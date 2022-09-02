//! The Conductor coordinates a set of Clients, scheduling requests (generated by the Workload),
//! and receiving replies (validated by the Workload).
//!
//! Replies from the cluster may arrive out-of-order; the Conductor reassembles them in the
//! correct order (by ascending op number) before passing them into the Workload.
const std = @import("std");
const assert = std.debug.assert;
const log = std.log.scoped(.test_conductor);

const vsr = @import("../vsr.zig");
const config = @import("../config.zig");
const IdPermutation = @import("id.zig").IdPermutation;
const MessagePool = @import("../message_pool.zig").MessagePool;
const Message = MessagePool.Message;

// TODO(zig) This won't be necessary in Zig 0.10.
const PriorityQueue = @import("./priority_queue.zig").PriorityQueue;

/// Both messages belong to the Conductor's `MessagePool`.
const PendingReply = struct {
    client_index: usize,
    request: *Message,
    reply: *Message,

    /// `PendingReply`s are ordered by ascending reply op.
    fn compare(context: void, a: PendingReply, b: PendingReply) std.math.Order {
        _ = context;
        return std.math.order(a.reply.header.op, b.reply.header.op);
    }
};

const PendingReplyQueue = PriorityQueue(PendingReply, void, PendingReply.compare);

pub fn ConductorType(
    comptime Client: type,
    comptime MessageBus: type,
    comptime StateMachine: type,
    comptime Workload: type,
) type {
    return struct {
        const Self = @This();

        /// Reply messages (from cluster to client) may be reordered during transit.
        /// The Conductor must reassemble them in the original order (ascending op/commit
        /// number) before handing them off to the Workload for verification.
        ///
        /// `Conduction.stalled_queue` hold replies (and corresponding requests) that are
        /// waiting to be processed.
        pub const stalled_queue_capacity = config.clients_max * config.client_request_queue_max * 2;

        random: std.rand.Random,
        workload: *Workload,
        options: Options,
        client_id_permutation: IdPermutation,

        clients: []Client,
        client_pools: []MessagePool,
        message_pool: MessagePool,

        /// The next op to be verified.
        /// Starts at 1, because op=0 is the root.
        stalled_op: u64 = 1,

        /// The list of messages waiting to be verified (the reply for a lower op has not yet arrived).
        /// Includes `register` messages.
        stalled_queue: PendingReplyQueue,

        /// Total number of messages sent, including those that have not been delivered.
        /// Does not include `register` messages.
        requests_sent: usize = 0,

        idle: bool = false,

        const Options = struct {
            cluster: u32,
            replica_count: u8,
            client_count: u8,
            message_bus_options: MessageBus.Options,

            /// The total number of requests to send. Does not count `register` messages.
            requests_max: usize,

            request_probability: u8, // percent
            idle_on_probability: u8, // percent
            idle_off_probability: u8, // percent
        };

        pub fn init(
            allocator: std.mem.Allocator,
            random: std.rand.Random,
            workload: *Workload,
            options: Options,
        ) !Self {
            assert(options.replica_count >= 1);
            assert(options.replica_count <= 6);
            assert(options.client_count > 0);
            assert(options.client_count * 2 < stalled_queue_capacity);
            assert(options.requests_max > 0);

            assert(options.request_probability > 0);
            assert(options.request_probability <= 100);
            assert(options.idle_on_probability <= 100);
            assert(options.idle_off_probability > 0);
            assert(options.idle_off_probability <= 100);

            // *2 for PendingReply.request and PendingReply.reply.
            var message_pool = try MessagePool.init_capacity(allocator, stalled_queue_capacity * 2);
            errdefer message_pool.deinit(allocator);

            var client_pools = try allocator.alloc(MessagePool, options.client_count);
            errdefer allocator.free(client_pools);

            for (client_pools) |*pool, i| {
                errdefer for (client_pools[0..i]) |*p| p.deinit(allocator);
                pool.* = try MessagePool.init(allocator, .client);
            }
            errdefer for (client_pools) |*p| p.deinit(allocator);

            var clients = try allocator.alloc(Client, options.client_count);
            errdefer allocator.free(clients);

            // Always use UUIDs because the simulator network expects client ids to never collide
            // with replica indices.
            const client_id_permutation = IdPermutation{ .random = random.int(u64) };

            for (clients) |*client, i| {
                errdefer for (clients[0..i]) |*c| c.deinit(allocator);
                client.* = try Client.init(
                    allocator,
                    // +1 so that index=0 is encoded as a valid id.
                    client_id_permutation.encode(i + 1),
                    options.cluster,
                    options.replica_count,
                    &client_pools[i],
                    options.message_bus_options,
                );
                client.on_reply_callback = on_reply;
            }
            errdefer for (clients) |*c| c.deinit(allocator);

            var stalled_queue = PendingReplyQueue.init(allocator, {});
            errdefer stalled_queue.deinit();
            try stalled_queue.ensureTotalCapacity(stalled_queue_capacity);

            return Self{
                .random = random,
                .workload = workload,
                .options = options,
                .client_id_permutation = client_id_permutation,
                .clients = clients,
                .client_pools = client_pools,
                .message_pool = message_pool,
                .stalled_queue = stalled_queue,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            while (self.stalled_queue.removeOrNull()) |pending| {
                self.message_pool.unref(pending.request);
                self.message_pool.unref(pending.reply);
            }
            self.stalled_queue.deinit();
            self.message_pool.deinit(allocator);

            for (self.clients) |*client| client.deinit(allocator);
            allocator.free(self.clients);

            for (self.client_pools) |*pool| pool.deinit(allocator);
            allocator.free(self.client_pools);
        }

        /// The conductor is "done" when it has delivered `requests_max` requests and received all
        /// replies.
        pub fn done(self: *const Self) bool {
            assert(self.requests_sent <= self.options.requests_max);
            if (self.requests_sent < self.options.requests_max) return false;

            for (self.clients) |*client| {
                if (client.request_queue.count > 0) return false;
            }

            return true;
        }

        pub fn tick(self: *Self) void {
            for (self.clients) |*client| {
                // TODO(zig) Move this into init when `@returnAddress()` is available. It only needs to
                // be set once, it just requires a stable pointer to the Conductor.
                client.on_reply_context = self;
                client.tick();
            }

            if (self.done()) return;

            // Try to pick a client & queue a request.

            if (self.idle) {
                if (chance(self.random, self.options.idle_off_probability)) self.idle = false;
            } else {
                if (chance(self.random, self.options.idle_on_probability)) self.idle = true;
            }
            if (self.idle) return;
            if (!chance(self.random, self.options.request_probability)) return;

            if (self.requests_sent == self.options.requests_max) return;
            assert(self.requests_sent < self.options.requests_max);

            // Messages aren't added to `stalled_queue` until a reply arrives.
            // Before sending a new message, make sure there will definitely be room for it.
            var reserved: usize = 0;
            for (self.clients) |*c| {
                // Count the number of clients that are still waiting for a `register` to complete,
                // since they may start one at any time.
                reserved += @boolToInt(c.session == 0);
                // Count the number of requests queued.
                reserved += c.request_queue.count;
            }

            // +1 for the potential request — is there room in our queue?
            if (self.stalled_queue.len + reserved + 1 > stalled_queue_capacity) return;

            const client_index = self.random.uintLessThanBiased(usize, self.clients.len);
            var client = &self.clients[client_index];

            // Check for space in the client's own request queue.
            if (client.request_queue.count + 1 > config.client_request_queue_max) return;

            var request_message = client.get_message();
            defer client.unref(request_message);

            const request_metadata = self.workload.build_request(
                client_index,
                @alignCast(@alignOf(vsr.Header), request_message.buffer[@sizeOf(vsr.Header)..config.message_size_max]),
            );
            assert(request_metadata.size <= config.message_size_max - @sizeOf(vsr.Header));

            client.request(
                0,
                request_callback,
                request_metadata.operation,
                request_message,
                request_metadata.size,
            );
            // Since we already checked the client's request queue for free space, `client.request()`
            // should always queue the request.
            assert(request_message == client.request_queue.tail_ptr().?.message);
            assert(request_message.header.client == client.id);
            assert(request_message.header.request == client.request_number - 1);
            assert(request_message.header.size == @sizeOf(vsr.Header) + request_metadata.size);
            assert(request_message.header.operation.cast(StateMachine) == request_metadata.operation);

            self.requests_sent += 1;
            assert(self.requests_sent <= self.options.requests_max);
        }

        /// The `request_callback` is not used. The Conductor needs access to the request/reply
        /// Messages to process them in the proper (op) order.
        ///
        /// See `on_reply`.
        fn request_callback(
            user_data: u128,
            operation: StateMachine.Operation,
            result: Client.Error![]const u8,
        ) void {
            _ = user_data;
            _ = operation;
            _ = result catch |err| switch (err) {
                error.TooManyOutstandingRequests => unreachable,
            };
        }

        fn on_reply(context: ?*anyopaque, client: *Client, request_message: *Message, reply_message: *Message) void {
            const self = @ptrCast(*Self, @alignCast(@alignOf(*Self), context.?));
            assert(reply_message.header.cluster == self.options.cluster);
            assert(reply_message.header.invalid() == null);
            assert(reply_message.header.client == client.id);
            assert(reply_message.header.request == request_message.header.request);
            assert(reply_message.header.op >= self.stalled_op);
            assert(reply_message.header.command == .reply);
            assert(reply_message.header.operation == request_message.header.operation);

            const client_id = reply_message.header.client;
            // -1 because id=0 is not valid, so index=0→id=1.
            const client_index = @intCast(usize, self.client_id_permutation.decode(client_id) - 1);
            self.stalled_queue.add(.{
                .client_index = client_index,
                .request = self.clone_message(request_message),
                .reply = self.clone_message(reply_message),
            }) catch unreachable;

            if (reply_message.header.op == self.stalled_op) {
                self.consume_stalled_replies();
            }
        }

        /// Copy the message from a Client's MessagePool to the Conductor's MessagePool.
        ///
        /// The client has a finite amount of messages in its pool, and the Conductor needs to hold
        /// onto requests/replies until all preceeding requests/replies have arrived.
        ///
        /// Returns the Conductor's message.
        fn clone_message(self: *Self, message_client: *const Message) *Message {
            const message_conductor = self.message_pool.get_message();
            std.mem.copy(u8, message_conductor.buffer, message_client.buffer);
            return message_conductor;
        }

        fn consume_stalled_replies(self: *Self) void {
            assert(self.stalled_queue.len > 0);
            assert(self.stalled_queue.len <= stalled_queue_capacity);
            while (self.stalled_queue.peek()) |head| {
                assert(head.reply.header.op >= self.stalled_op);
                if (head.reply.header.op != self.stalled_op) break;

                const commit = self.stalled_queue.remove();
                defer self.message_pool.unref(commit.reply);
                defer self.message_pool.unref(commit.request);

                assert(commit.reply.references == 1);
                assert(commit.reply.header.command == .reply);
                assert(commit.reply.header.client == self.clients[commit.client_index].id);
                assert(commit.reply.header.request == commit.request.header.request);
                assert(commit.reply.header.op == self.stalled_op);
                assert(commit.reply.header.operation == commit.request.header.operation);

                assert(commit.request.references == 1);
                assert(commit.request.header.command == .request);
                assert(commit.request.header.client == self.clients[commit.client_index].id);

                log.debug("consume_stalled_replies: op={} operation={} client={} request={}", .{
                    commit.reply.header.op,
                    commit.reply.header.operation,
                    commit.request.header.client,
                    commit.request.header.request,
                });

                if (commit.request.header.operation != .register) {
                    self.workload.on_reply(
                        commit.client_index,
                        commit.reply.header.operation,
                        commit.reply.header.timestamp,
                        commit.request.body(),
                        commit.reply.body(),
                    );
                }
                self.stalled_op += 1;
            }
        }
    };
}

/// Returns true, `p` percent of the time, else false.
fn chance(random: std.rand.Random, p: u8) bool {
    assert(p <= 100);
    return random.uintLessThanBiased(u8, 100) < p;
}
