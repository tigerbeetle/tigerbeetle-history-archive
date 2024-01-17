//! Compaction moves or merges a table's values from the previous level.
//!
//! Each Compaction is paced to run in an arbitrary amount of beats, by the forest.
//!
//!
//! Compaction overview:
//!
//! 1. Given:
//!
//!   - levels A and B, where A+1=B
//!   - a single table in level A ("table A")
//!   - all tables from level B which intersect table A's key range ("tables B")
//!     (This can include anything between 0 tables and all of level B's tables.)
//!
//! 2. If table A's key range is disjoint from the keys in level B, move table A into level B.
//!    All done! (But if the key ranges intersect, jump to step 3).
//! FIXME: Update
//! 3. Create an iterator from the sort-merge of table A and the concatenation of tables B.
//!    If the same key exists in level A and B, take A's and discard B's. †
//!
//! 4. Write the sort-merge iterator into a sequence of new tables on disk.
//!
//! 5. Update the input tables in the Manifest with their new `snapshot_max` so that they become
//!    invisible to subsequent read transactions.
//!
//! 6. Insert the new level-B tables into the Manifest.
//!
//! † When A's value is a tombstone, there is a special case for garbage collection. When either:
//! * level B is the final level, or
//! * A's key does not exist in B or any deeper level,
//! then the tombstone is omitted from the compacted output, see: `compaction_must_drop_tombstones`.
//!
const std = @import("std");
const mem = std.mem;
const math = std.math;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

const log = std.log.scoped(.compaction);
const tracer = @import("../tracer.zig");

const constants = @import("../constants.zig");

const stdx = @import("../stdx.zig");
const GridType = @import("../vsr/grid.zig").GridType;
const BlockPtr = @import("../vsr/grid.zig").BlockPtr;
const BlockPtrConst = @import("../vsr/grid.zig").BlockPtrConst;
const allocate_block = @import("../vsr/grid.zig").allocate_block;
const TableInfoType = @import("manifest.zig").TreeTableInfoType;
const ManifestType = @import("manifest.zig").ManifestType;
const schema = @import("schema.zig");

/// The upper-bound count of input tables to a single tree's compaction.
///
/// - +1 from level A.
/// - +lsm_growth_factor from level B. The A-input table cannot overlap with an extra B-input table
///   because input table selection is least-overlap. If the input table overlaps on one or both
///   edges, there must be another table with less overlap to select.
pub const compaction_tables_input_max = 1 + constants.lsm_growth_factor;

/// The upper-bound count of output tables from a single tree's compaction.
/// In the "worst" case, no keys are overwritten/merged, and no tombstones are dropped.
pub const compaction_tables_output_max = compaction_tables_input_max;

/// Information used when scheduling compactions. Kept unspecalized to make the forest
/// code easier.
pub const CompactionInfo = struct {
    /// How many values, across all tables, need to be processed.
    value_count: usize,
};

pub const CompactionBlocks = struct {
    /// FIXME: Input index blocks shared... Not sure of consequences yet.
    input_index_blocks: []BlockPtr,

    /// For each split, for each input level, we have a buffer of blocks.
    input_data_blocks: [2][2][]BlockPtr,

    /// For each split we have a buffer of blocks.
    output_data_blocks: [2][]BlockPtr,
};

const BlipCallback = *const fn (*anyopaque, ?bool, ?bool) void;

pub fn CompactionInterfaceType(comptime Grid: type, comptime tree_infos: anytype) type {
    return struct {
        const Dispatcher = T: {
            var type_info = @typeInfo(union(enum) {});

            // Union fields for each compaction type.
            for (tree_infos) |tree_info| {
                const Compaction = tree_info.Tree.Compaction;
                const type_name = @typeName(Compaction);

                for (type_info.Union.fields) |field| {
                    if (std.mem.eql(u8, field.name, type_name)) {
                        break;
                    }
                } else {
                    type_info.Union.fields = type_info.Union.fields ++ [_]std.builtin.Type.UnionField{.{
                        .name = type_name,
                        .type = *Compaction,
                        .alignment = @alignOf(*Compaction),
                    }};
                }
            }

            // We need a tagged union for dynamic dispatching.
            type_info.Union.tag_type = blk: {
                const union_fields = type_info.Union.fields;
                var tag_fields: [union_fields.len]std.builtin.Type.EnumField =
                    undefined;
                for (&tag_fields, union_fields, 0..) |*tag_field, union_field, i| {
                    tag_field.* = .{
                        .name = union_field.name,
                        .value = i,
                    };
                }

                break :blk @Type(.{ .Enum = .{
                    .tag_type = std.math.IntFittingRange(0, tag_fields.len - 1),
                    .fields = &tag_fields,
                    .decls = &.{},
                    .is_exhaustive = true,
                } });
            };

            break :T @Type(type_info);
        };

        const Self = @This();

        dispatcher: Dispatcher,
        info: CompactionInfo,

        pub fn init(info: CompactionInfo, compaction: anytype) @This() {
            const Compaction = @TypeOf(compaction.*);
            const type_name = @typeName(Compaction);

            return .{
                .info = info,
                .dispatcher = @unionInit(Dispatcher, type_name, compaction),
            };
        }

        pub fn bar_setup_budget(self: *const Self, beat_budget: u64, output_index_blocks: []BlockPtr) void {
            return switch (self.dispatcher) {
                inline else => |compaction_impl| compaction_impl.bar_setup_budget(beat_budget, output_index_blocks),
            };
        }

        pub fn beat_grid_acquire(self: *Self) void {
            return switch (self.dispatcher) {
                inline else => |compaction_impl| compaction_impl.beat_grid_acquire(),
            };
        }

        pub fn beat_blocks_assign(self: *Self, blocks: CompactionBlocks, grid_reads: []Grid.FatRead, grid_writes: []Grid.FatWrite) void {
            return switch (self.dispatcher) {
                inline else => |compaction_impl| compaction_impl.beat_blocks_assign(blocks, grid_reads, grid_writes),
            };
        }

        // FIXME: Very unhappy with the callback style here!
        pub fn blip_read(self: *Self, callback: BlipCallback) void {
            return switch (self.dispatcher) {
                inline else => |compaction_impl| compaction_impl.blip_read(callback, self),
            };
        }

        pub fn blip_merge(self: *Self, callback: BlipCallback) void {
            return switch (self.dispatcher) {
                inline else => |compaction_impl| compaction_impl.blip_merge(callback, self),
            };
        }

        pub fn blip_write(self: *Self, callback: BlipCallback) void {
            return switch (self.dispatcher) {
                inline else => |compaction_impl| compaction_impl.blip_write(callback, self),
            };
        }

        // pub fn beat_blocks_release(self: *Self) void {
        //     return switch (self.dispatcher) {
        //         inline else => |compaction_impl| compaction_impl.beat_blocks_release(),
        //     };
        // }

        pub fn beat_grid_forfeit(self: *Self) void {
            return switch (self.dispatcher) {
                inline else => |compaction_impl| compaction_impl.beat_grid_forfeit(),
            };
        }
    };
}

pub fn CompactionType(
    comptime Table: type,
    comptime Tree: type,
    comptime Storage: type,
) type {
    return struct {
        const Compaction = @This();

        const Grid = GridType(Storage);
        pub const Tree_ = Tree;

        const Manifest = ManifestType(Table, Storage);
        const TableInfo = TableInfoType(Table);
        const TableInfoReference = Manifest.TableInfoReference;
        const CompactionRange = Manifest.CompactionRange;

        const Key = Table.Key;
        const Value = Table.Value;
        const key_from_value = Table.key_from_value;
        const tombstone = Table.tombstone;

        const TableInfoA = union(enum) {
            immutable: *Tree.TableMemory,
            disk: TableInfoReference,
        };

        const InputLevel = enum {
            a,
            b,
        };

        const ValueBlockIterator = struct {
            values: []const Value,
            source_index: usize = 0,

            // FIXME: Useless init fn. Maybe assert stuff idk.
            fn init(values: []const Value, source_index: usize) ValueBlockIterator {
                return .{
                    .values = values,
                    .source_index = source_index,
                };
            }

            pub fn next(self: *ValueBlockIterator) ?Table.Value {
                if (self.source_index == self.values.len) {
                    return null;
                }

                const val = self.values[self.source_index];
                self.source_index += 1;

                return val;
            }

            pub fn count(self: *const ValueBlockIterator) usize {
                return self.values.len;
            }

            pub fn remaining(self: *const ValueBlockIterator) usize {
                return self.values.len - self.source_index;
            }

            pub fn copy(self: *ValueBlockIterator, table_builder: *Table.Builder) void {
                assert(table_builder.value_count < Table.layout.block_value_count_max);

                const values_in = self.values[self.source_index..];
                const values_out = table_builder.data_block_values();

                var values_out_index = table_builder.value_count;

                assert(values_in.len > 0);

                const len = @min(values_in.len, values_out.len - values_out_index);
                assert(len > 0);
                stdx.copy_disjoint(
                    .exact,
                    Value,
                    values_out[values_out_index..][0..len],
                    values_in[0..len],
                );

                self.source_index += len;
                table_builder.value_count += @as(u32, @intCast(len));
            }
        };

        const UnifiedIterator = struct {
            const Dispatcher = union(enum) {
                value_block_iterator: ValueBlockIterator,
                table_memory_iterator: Tree.TableMemory.Iterator,
            };

            dispatcher: Dispatcher,

            // FIXME: Do we want this iterator to return ptrs rather?
            pub fn next(self: *UnifiedIterator) ?Table.Value {
                return switch (self.dispatcher) {
                    inline else => |*iterator_impl| iterator_impl.next(),
                };
            }

            pub fn count(self: *const UnifiedIterator) usize {
                return switch (self.dispatcher) {
                    inline else => |iterator_impl| iterator_impl.count(),
                };
            }

            pub fn remaining(self: *const UnifiedIterator) usize {
                return switch (self.dispatcher) {
                    inline else => |iterator_impl| iterator_impl.remaining(),
                };
            }

            pub fn get_source_index(self: *const UnifiedIterator) usize {
                return switch (self.dispatcher) {
                    inline else => |iterator_impl| iterator_impl.source_index,
                };
            }

            /// We allow each underlying iterator type to implement its own optimized copy method.
            pub fn copy(self: *UnifiedIterator, table_builder: *Table.Builder) void {
                return switch (self.dispatcher) {
                    inline else => |*iterator_impl| iterator_impl.copy(table_builder),
                };
            }
        };

        const ValuesIn = struct {
            UnifiedIterator,
            UnifiedIterator,
        };

        const PipelineStage = enum { read, merge, write };

        const BarContext = struct {
            tree: *Tree,

            /// `op_min` is the first op/beat of this compaction's half-bar.
            /// `op_min` is used as a snapshot — the compaction's input tables must be visible
            /// to `op_min`.
            ///
            /// After this compaction finishes:
            /// - `op_min + half_bar_beat_count - 1` will be the input tables' snapshot_max.
            /// - `op_min + half_bar_beat_count` will be the output tables' snapshot_min.
            op_min: u64,

            /// Whether this compaction will use the move-table optimization.
            /// Specifically, this field is set to True if the optimal compaction
            /// table in level A can simply be moved to level B.
            move_table: bool,

            table_info_a: TableInfoA,
            range_b: CompactionRange,

            /// Levels may choose to drop tombstones if keys aren't included in the lower levels.
            /// This invariant is always true for the last level as it doesn't have any lower ones.
            drop_tombstones: bool,

            /// Number of beats we should aim to finish this compaction in. It might be fewer, but it'll
            /// never be more.
            beat_budget: ?u64,
            total_value_count: u64,
            per_beat_input_goal: u64 = 0,

            /// We have to track our input iterator positions across bars - FIXME: explain why
            current_block_a_idx: usize = 0,
            current_block_b_idx: usize = 0,

            /// Track our index position.
            current_table_a_value_block_index: usize = 0,
            current_table_b_index_block_index: usize = 0,
            current_table_b_value_block_index: usize = 0,

            /// At least 1 output index block needs to span beat boundaries, otherwise it wouldn't be
            /// possible to pace at a more granular level than tables.
            output_index_blocks: []BlockPtr,
            current_index_block: usize = 0, // FIXME: assert less than len above in places

            /// Manifest log appends are queued up until `finish()` is explicitly called to ensure
            /// they are applied deterministically relative to other concurrent compactions.
            manifest_entries: stdx.BoundedArray(struct {
                operation: enum {
                    insert_to_level_b,
                    move_to_level_b,
                },
                table: TableInfo,
            }, manifest_entries_max: {
                // Worst-case manifest updates:
                // See docs/internals/lsm.md "Compaction Table Overlap" for more detail.
                var count = 0;

                count += constants.lsm_growth_factor + 1; // Insert the output tables to level B.
                // (In the move-table case, only a single TableInfo is inserted, and none are updated.)
                break :manifest_entries_max count;
            }) = .{},

            table_builder: Table.Builder = .{},
        };

        const BeatContext = struct {
            const ReadContext = struct {
                callback: BlipCallback,
                ptr: *anyopaque,
                pending_index_reads: usize = 0,
                pending_data_reads: usize = 0,
                next_tick: Grid.NextTick = undefined,
                timer: std.time.Timer,
                timer_read: usize = 0,
            };
            const MergeContext = struct {
                callback: BlipCallback,
                ptr: *anyopaque,
                current_block_a: usize = 0,
                current_block_b: usize = 0,
                current_output_data_block: usize = 0,
                next_tick: Grid.NextTick = undefined,
                timer: std.time.Timer,
            };
            const WriteContext = struct {
                callback: BlipCallback,
                ptr: *anyopaque,
                data_blocks_count: usize = 0,
                pending_writes: usize = 0,
                next_tick: Grid.NextTick = undefined,
                timer: std.time.Timer,
            };

            grid_reservation: ?Grid.Reservation,

            // Index blocks don't have the same data dependency on cpu that data blocks do.
            // FIXME: undefined!
            blocks: CompactionBlocks = undefined,

            grid_reads: []Grid.FatRead,
            grid_writes: []Grid.FatWrite,

            input_values_processed: u64 = 0,

            // Unlike other places where we can use a single state enum, a single Compaction
            // instance is _expected_ to be reading, writing and merging all at once. These
            // are not disjoint states!
            //
            // {read,merge,write} are considered inactive if their context is null.
            read: ?ReadContext = null,
            read_current_split: usize = 0,

            merge: ?MergeContext = null,
            merge_current_split: usize = 0,

            write: ?WriteContext = null,
            write_current_split: usize = 0,

            fn activate_and_assert(self: *BeatContext, stage: PipelineStage, callback: BlipCallback, ptr: *anyopaque) void {
                switch (stage) {
                    .read => {
                        assert(self.read == null);
                        assert(self.merge == null or self.merge_current_split != self.read_current_split);
                        assert(self.write == null or self.read_current_split == self.write_current_split);

                        self.read = .{
                            .callback = callback,
                            .ptr = ptr,
                            .timer = std.time.Timer.start() catch unreachable,
                        };
                        self.read.?.timer.reset();
                    },
                    .merge => {
                        assert(self.merge == null);
                        assert(self.read == null or self.merge_current_split != self.read_current_split);
                        assert(self.write == null or self.merge_current_split != self.write_current_split);

                        self.merge = .{
                            .callback = callback,
                            .ptr = ptr,
                            .timer = std.time.Timer.start() catch unreachable,
                        };
                        self.merge.?.timer.reset();
                    },
                    .write => {
                        assert(self.write == null);
                        assert(self.merge == null or self.merge_current_split != self.write_current_split);
                        assert(self.read == null or self.read_current_split == self.write_current_split);

                        self.write = .{
                            .callback = callback,
                            .ptr = ptr,
                            .timer = std.time.Timer.start() catch unreachable,
                        };
                        self.write.?.timer.reset();
                    },
                }
            }

            fn deactivate_and_assert_and_callback(self: *BeatContext, stage: PipelineStage, arg1: ?bool, arg2: ?bool) void {
                switch (stage) {
                    .read => {
                        assert(self.read != null);
                        assert(self.read.?.pending_index_reads == 0);
                        assert(self.read.?.pending_data_reads == 0);

                        const callback = self.read.?.callback;
                        const ptr = self.read.?.ptr;

                        self.read_current_split = (self.read_current_split + 1) % 2;
                        self.read = null;

                        callback(ptr, arg1, arg2);
                    },
                    .merge => {
                        assert(self.merge != null);

                        const callback = self.merge.?.callback;
                        const ptr = self.merge.?.ptr;

                        self.merge_current_split = (self.merge_current_split + 1) % 2;
                        self.merge = null;

                        callback(ptr, arg1, arg2);
                    },
                    .write => {
                        assert(self.write != null);
                        assert(self.write.?.pending_writes == 0);

                        const callback = self.write.?.callback;
                        const ptr = self.write.?.ptr;

                        self.write_current_split = (self.write_current_split + 1) % 2;
                        self.write = null;

                        callback(ptr, arg1, arg2);
                    },
                }
            }

            fn assert_all_inactive(self: *BeatContext) void {
                assert(self.read == null);
                assert(self.merge == null);
                assert(self.write == null);
            }
        };

        // Passed by `init`.
        tree_config: Tree.Config,
        level_b: u8,
        grid: *Grid,

        // Allocated during `init`.
        last_keys_in: [2]?Key = .{ null, null },

        // Populated by {bar,beat}_setup.
        bar_context: ?BarContext,
        beat_context: ?BeatContext,

        pub fn init(tree_config: Tree.Config, grid: *Grid, level_b: u8) !Compaction {
            assert(level_b < constants.lsm_levels);

            return Compaction{
                .tree_config = tree_config,
                .grid = grid,
                .level_b = level_b,

                .bar_context = null,
                .beat_context = null,
            };
        }

        pub fn deinit(compaction: *Compaction) void {
            _ = compaction;
        }

        pub fn reset(compaction: *Compaction) void {
            // FIXME: Ensure blocks are released... Also what if bar_context is null.
            compaction.bar_context.?.table_builder.reset();

            compaction.* = .{
                .tree_config = compaction.tree_config,
                .grid = compaction.grid,
                .level_b = compaction.level_b,

                .bar_context = null,
                .beat_context = null,
            };
        }

        /// Called per bar
        /// Perform the bar-wise setup, and return information that the forest can use for scheduling.
        /// beat_budget is the number of beats that this compaction will have available to do its work.
        /// A compaction may be done before beat_budget, if eg tables are mostly empty.
        /// Output index blocks are special, and are allocated at a bar level unlike all the other blocks
        /// which are done at a beat level. This is because while we can ensure we fill a data block, index
        /// blocks are too infrequent (one per table) to divide compaction by.
        /// Returns null if there's no compaction work, otherwise returns the compaction work that needs to be done.
        /// move_table is a semi-special case - a non-null value is returned because we still need to do manifest
        /// updates in bar_finish, but we don't need to actually _do_ anything else.
        pub fn bar_setup(compaction: *Compaction, tree: *Tree, op: u64) ?CompactionInfo {
            assert(compaction.bar_context == null);
            assert(compaction.beat_context == null);

            // std.log.info("bar_setup running for compaction: {s} into level: {}", .{ compaction.tree_config.name, compaction.level_b });

            // FIXME values_count vs value_count
            var total_value_count: usize = 0;

            // level_b 0 is special; unlike all the others which have a level_a on disk, level 0's
            // corresponding level_a comes from the immutable table. This means that blip_read will
            // effectively be a noop, and that the minimum input blocks are lowered by one.
            // TODO: Don't know if we want to make use of that just yet.
            if (compaction.level_b == 0) {
                // Do not start compaction if the immutable table does not require compaction.
                if (tree.table_immutable.mutability.immutable.flushed) {
                    return null;
                }

                const values = tree.table_immutable.values_used();
                const values_count = values.len;
                assert(values_count > 0);

                const range_b = tree.manifest.immutable_table_compaction_range(
                    tree.table_immutable.key_min(),
                    tree.table_immutable.key_max(),
                );

                // +1 to count the immutable table (level A).
                assert(range_b.tables.count() + 1 <= compaction_tables_input_max);
                assert(range_b.key_min <= tree.table_immutable.key_min());
                assert(tree.table_immutable.key_max() <= range_b.key_max);

                // FIXME: Hmm about op_min here
                log.info("{s}: compacting immutable table to level 0 " ++
                    "(snapshot_min={d} compaction.op_min={d} table_count={d} values={d})", .{
                    tree.config.name,
                    tree.table_immutable.mutability.immutable.snapshot_min,
                    op,
                    range_b.tables.count() + 1,
                    values_count,
                });

                total_value_count += values_count;
                for (range_b.tables.const_slice()) |*table| {
                    // FIXME: We don't deal with this yet.
                    // if (table.table_info.value_count > 0) assert(false);
                    total_value_count += table.table_info.value_count;
                }

                compaction.bar_context = .{
                    .tree = tree,
                    .op_min = op,

                    .move_table = false,
                    .table_info_a = .{ .immutable = &tree.table_immutable },
                    .range_b = range_b,
                    .drop_tombstones = tree.manifest.compaction_must_drop_tombstones(
                        compaction.level_b,
                        range_b,
                    ),

                    .total_value_count = total_value_count,

                    // FIXME: Don't like this! How to do it better?
                    .output_index_blocks = undefined,
                    .beat_budget = null,
                };
            } else {
                // Do not start compaction if level A does not require compaction.
                const level_a = compaction.level_b - 1;

                const table_range = tree.manifest.compaction_table(level_a) orelse return null;
                const table_a = table_range.table_a.table_info;
                const range_b = table_range.range_b;

                assert(range_b.tables.count() + 1 <= compaction_tables_input_max);
                assert(table_a.key_min <= table_a.key_max);
                assert(range_b.key_min <= table_a.key_min);
                assert(table_a.key_max <= range_b.key_max);

                log.info("{s}: compacting {d} tables from level {d} to level {d}", .{
                    tree.config.name,
                    range_b.tables.count() + 1,
                    level_a,
                    compaction.level_b,
                });

                const perform_move_table = range_b.tables.empty();

                total_value_count += table_a.value_count;
                for (range_b.tables.const_slice()) |*table| total_value_count += table.table_info.value_count;

                compaction.bar_context = .{
                    .tree = tree,
                    .op_min = op,

                    .move_table = perform_move_table,
                    .table_info_a = .{ .disk = table_range.table_a },
                    .range_b = range_b,
                    .drop_tombstones = tree.manifest.compaction_must_drop_tombstones(
                        compaction.level_b,
                        range_b,
                    ),

                    .total_value_count = total_value_count,

                    // FIXME: Don't like this! How to do it better?
                    .output_index_blocks = undefined,
                    .beat_budget = null,
                };

                if (perform_move_table) {
                    compaction.move_table();
                    return null;
                }
            }

            assert(compaction.bar_context.?.drop_tombstones or compaction.level_b < constants.lsm_levels - 1);

            return .{
                .value_count = total_value_count,
            };
        }

        /// Setup the per beat budget, as well as the output index block. Done in a separate step to bar_setup()
        /// since the forest requires information from that step to calculate how it should split the work, and
        /// if there's move table, output_index_blocks must be len 0.
        // Minimum of 1, max lsm_growth_factor+1 of output_index_blocks.
        pub fn bar_setup_budget(compaction: *Compaction, beat_budget: u64, output_index_blocks: []BlockPtr) void {
            assert(beat_budget <= constants.lsm_batch_multiple);
            assert(compaction.bar_context != null);
            assert(compaction.beat_context == null);

            const bar_context = &compaction.bar_context.?;

            assert(bar_context.per_beat_input_goal == 0);

            // FIXME: Actually move this calc into beat_grid_acquire, and subtract the values we've already done from it.
            // This way we self correct our pacing!
            bar_context.per_beat_input_goal = stdx.div_ceil(bar_context.total_value_count, beat_budget);
            bar_context.output_index_blocks = output_index_blocks;

            if (bar_context.move_table) {
                // FIXME: Asserts here
                // assert(output_index_blocks.len == 0);
                // assert(compaction.bar_context.?.per_beat_input_goal == 0);
            } else {
                assert(output_index_blocks.len > 0);
                assert(bar_context.per_beat_input_goal > 0);
            }

            // FIXME: Ok, so this gets set once, but we do an extra data block. What we should do is recalculate this dynamically after each beat, to better spread
            // the work out....
            std.log.info("Total: {} per beat goal: {}", .{ bar_context.total_value_count, bar_context.per_beat_input_goal });
        }

        /// Called per beat
        pub fn beat_grid_acquire(
            compaction: *Compaction,
        ) void {
            assert(compaction.bar_context != null);
            assert(compaction.beat_context == null);

            const bar_context = &compaction.bar_context.?;
            assert(bar_context.per_beat_input_goal > 0);

            // If we're move_table, the calling code should ensure beat_setup and blip code are never called.
            assert(!bar_context.move_table);

            // Reserve enough blocks to write our output tables in the semi-worst case, where:
            // - no tombstones are dropped,
            // - no values are overwritten,
            // - and we know exact input value counts, so table fullness *is* accounted for.
            //
            // We must reserve before doing any async work so that the block acquisition order
            // is deterministic (relative to other concurrent compactions).
            //
            // We also need to account for index blocks.
            // FIXME: What if we have partially filled index block from previous beat, that we now fill - do we need a +1?
            const value_blocks_per_beat = stdx.div_ceil(bar_context.per_beat_input_goal, Table.layout.block_value_count_max);
            const index_blocks_per_beat = stdx.div_ceil(value_blocks_per_beat, Table.data_block_count_max);
            const total_blocks_per_beat = index_blocks_per_beat + value_blocks_per_beat;

            // TODO The replica must stop accepting requests if it runs out of blocks/capacity,
            // rather than panicking here.
            // (actually, we want to still panic but with something nicer like vsr.fail)
            const grid_reservation = compaction.grid.reserve(total_blocks_per_beat).?;

            compaction.beat_context = .{
                .grid_reservation = grid_reservation,
                .grid_reads = undefined,
                .grid_writes = undefined,
            };
        }

        pub fn beat_blocks_assign(compaction: *Compaction, blocks: CompactionBlocks, grid_reads: []Grid.FatRead, grid_writes: []Grid.FatWrite) void {
            assert(compaction.bar_context != null);
            assert(compaction.beat_context != null);

            const beat_context = &compaction.beat_context.?;
            const bar_context = &compaction.bar_context.?;

            beat_context.grid_reads = grid_reads;
            beat_context.grid_writes = grid_writes;
            beat_context.blocks = blocks;

            // FIXME: Not sure best way to handle this. Blocks are identical for r/w but read / writes arent'.
            assert(
                blocks.input_index_blocks.len + blocks.input_data_blocks[0][0].len + blocks.input_data_blocks[0][1].len + blocks.input_data_blocks[1][0].len + blocks.input_data_blocks[1][1].len <= grid_reads.len,
            );
            assert(bar_context.output_index_blocks.len + blocks.output_data_blocks[0].len + blocks.output_data_blocks[1].len <= grid_writes.len);
        }

        // Our blip pipeline is 3 stages long, and split into Read, Cpu and Write. Within a single compaction,
        // the pipeline looks something like:
        // ---------------------------------------------------------------------------------------------------------
        // | Ra₀    | Ca₀     | Wa₀     | Ra₁     | Rb₁     | Cb₁     | Wb₁           |            |
        // ---------------------------------------------------------------------------------------------------------
        // |       | Ra₁     | Ca₁     | Wa₁     | ---      | Rb₀     | Cb₀           | Wb₀           |
        // ---------------------------------------------------------------------------------------------------------
        // |       |        | Ra₀     | Ca₀ → E | Wb₀       | ---     | Rb₁           |  Cb₁          | Wb₁
        // ---------------------------------------------------------------------------------------------------------
        //
        // ---------------------------------------------------------------------------------------------------------
        // | Ra₀    | Ca₀     | Wa₀     | Ra₁     | Ca₁     | Wa₁     | Rb₀           |            |
        // ---------------------------------------------------------------------------------------------------------
        // |       | Ra₁     | Ca₁     | Wa₁     | Rb₀      | Cb₀     | Wb₀           |             |
        // ---------------------------------------------------------------------------------------------------------
        // |       |        | Ra₀     | Ca₀ → E | Wb₀       | Rb₁     | Cb₁           |  Wb₁
        // ---------------------------------------------------------------------------------------------------------

        // Internally, we have a reentry (better name?) counter - the first time read() is called, it works on buffer set 0, the next
        // time on buffer set 1. Ditto for cpu and write.
        // read(0) means read into read buffer set 0, cpu(0) means cpu merge from read buffer set 0 to write buffer set 0,
        // write(0) means write from write buffer set 0.
        //
        // Another note: even without a threadpool, we can likely drive better performance by doubling up the stages. The reason
        // for this is that we expect cpu() to be quite a bit quicker than write().
        //
        // The forest is free to pipeline other compactions of other trees with us, so these pipeline tetris blocks slot together.
        // IO work is always submitted to the kernel (ie, io_uring_enter) _before_ entering cpu() - since for now, we're single threaded.
        // cpu() frees the input buffers it uses, and write() frees its buffers too, so the actual amount of memory required is
        // 2x read() since a read will be issued concurrently with a cpu()
        // 2x write() since a cpu will be writing into blocks that are then flushed by a read()

        /// Perform read IO to fill our input_index_blocks and input_data_blocks with as many blocks
        /// as we can, given their size, and where we are in the amount of work we need to do this beat.
        /// Once we've read in our index blocks, we'll know how many data blocks we have. We can use this to do
        /// more precise pacing than our worst cast in beat_setup(). Specifically, we know exactly how many
        /// input data blocks we'll need to work through, so use that as our pacing metric. This is fixed.
        /// The number of output data blocks / tables will vary based on tombstones and overwites.
        //
        /// FIXME: If we treated our buffers as circular, we could be more efficient when doing our reads: rather than just read for
        /// eg one side if we're exhausted, we could read for both...
        /// HOWEVER! This isn't needed to be correct.
        pub fn blip_read(compaction: *Compaction, callback: BlipCallback, ptr: *anyopaque) void {
            // FIXME: Is there a point to asserting != null if we use .? later?
            assert(compaction.bar_context != null);
            assert(compaction.beat_context != null);

            const bar_context = &compaction.bar_context.?;
            _ = bar_context;
            const beat_context = &compaction.beat_context.?;

            beat_context.activate_and_assert(.read, callback, ptr);

            // FIXME: Verify the value count matches the number of values we actually compact.
            compaction.blip_read_index();
        }

        fn blip_read_index(compaction: *Compaction) void {
            assert(compaction.bar_context != null);
            assert(compaction.beat_context != null);

            const bar_context = &compaction.bar_context.?;
            const beat_context = &compaction.beat_context.?;

            assert(beat_context.read != null);
            const read = &beat_context.read.?;

            // index_block_a will always point to input_index_blocks[0] (even though if our source is immutable this isn't needed! future optimization)
            // index_block_b will be the index block of the table we're currently merging with. This will start from the left and move to the right
            // (although we might want to be able to reverse that in future...)

            var read_target: usize = 0;
            switch (bar_context.table_info_a) {
                .disk => |table_ref| {
                    beat_context.grid_reads[read_target].target = compaction;
                    beat_context.grid_reads[read_target].hack = read_target;
                    compaction.grid.read_block(
                        .{ .from_local_or_global_storage = blip_read_index_callback },
                        &beat_context.grid_reads[read_target].read,
                        table_ref.table_info.address,
                        table_ref.table_info.checksum,
                        .{ .cache_read = true, .cache_write = true },
                    );
                    read.pending_index_reads += 1;
                },
                .immutable => {
                    // Immutable values come from the in memory immutable table - no need to do any reads.
                },
            }
            // ALWAYS increment read_target for now.
            read_target += 1;

            // FIXME: This of course assumes infinite buffer space :)
            for (bar_context.range_b.tables.const_slice()) |table_ref| {
                beat_context.grid_reads[read_target].target = compaction;
                beat_context.grid_reads[read_target].hack = read_target;
                compaction.grid.read_block(
                    .{ .from_local_or_global_storage = blip_read_index_callback },
                    &beat_context.grid_reads[read_target].read,
                    table_ref.table_info.address,
                    table_ref.table_info.checksum,
                    .{ .cache_read = true, .cache_write = true },
                );
                read.pending_index_reads += 1;
                read_target += 1;
            }

            // Either we have pending index reads, in which case blip_read_data gets called by
            // blip_read_index_callback once all reads are done, or we don't, in which case call it
            // here.
            if (read.pending_index_reads == 0) {
                return compaction.blip_read_data();
            }
        }

        fn blip_read_index_callback(grid_read: *Grid.Read, index_block: BlockPtrConst) void {
            const parent = @fieldParentPtr(Grid.FatRead, "read", grid_read);
            const compaction: *Compaction = @alignCast(@ptrCast(parent.target));
            assert(compaction.bar_context != null);
            assert(compaction.beat_context != null);
            assert(compaction.tree_config.id == schema.TableIndex.metadata(index_block).tree_id);

            const bar_context = &compaction.bar_context.?;
            _ = bar_context;
            const beat_context = &compaction.beat_context.?;

            // FIXME: Figure out where our read target should go in a better way...
            assert(beat_context.read != null);
            const read = &beat_context.read.?;

            read.pending_index_reads -= 1;
            read.timer_read += 1;

            const read_index = parent.hack;
            stdx.copy_disjoint(.exact, u8, beat_context.blocks.input_index_blocks[read_index], index_block);
            std.log.info("Copied index block to {}", .{read_index});

            // FIXME: Not sure if I like this too much. According to release_table_blocks, it'll only release at the end of the bar, so should be ok?
            // FIXME: It's critical to release blocks, so ensure this is done properly.
            compaction.release_table_blocks(index_block);

            if (read.pending_index_reads != 0) return;

            compaction.blip_read_data();
        }

        fn blip_read_data(compaction: *Compaction) void {
            assert(compaction.bar_context != null);
            assert(compaction.beat_context != null);

            const bar_context = &compaction.bar_context.?;
            const beat_context = &compaction.beat_context.?;

            assert(beat_context.read != null);
            const read = &beat_context.read.?;

            read.timer_read += 1;

            // Used to select the Grid.Read to use - so shared between table a and table b.
            var read_target: usize = 0;

            // Read data for table a - which we'll only have if we're coming from disk.
            if (bar_context.table_info_a == .disk) {
                const index_block = beat_context.blocks.input_index_blocks[0];
                const index_schema = schema.TableIndex.from(index_block);

                const data_blocks_used = index_schema.data_blocks_used(index_block);
                assert(bar_context.current_table_a_value_block_index < data_blocks_used);

                const data_block_addresses = index_schema.data_addresses_used(index_block);
                const data_block_checksums = index_schema.data_checksums_used(index_block);

                // Issue as many reads as we have space for.
                var data_blocks_read: usize = 0;

                while (data_blocks_read < beat_context.blocks.input_data_blocks[beat_context.read_current_split][0].len and data_blocks_read < data_blocks_used) {
                    beat_context.grid_reads[read_target].target = compaction;
                    beat_context.grid_reads[read_target].hack = read_target;
                    compaction.grid.read_block(
                        .{ .from_local_or_global_storage = blip_read_data_callback },
                        &beat_context.grid_reads[read_target].read,
                        data_block_addresses[data_blocks_read],
                        data_block_checksums[data_blocks_read].value,
                        .{ .cache_read = true, .cache_write = true },
                    );

                    read.pending_data_reads += 1;
                    read_target += 1;
                    data_blocks_read += 1;
                }
            }

            // Read data for our tables in range b.
            const table_b_count = bar_context.range_b.tables.count();

            var previous_schema: ?schema.TableIndex = null;
            outer: while (bar_context.current_table_b_index_block_index < table_b_count) : (bar_context.current_table_b_index_block_index += 1) {
                // The 1 + is to skip over the table a index block.
                const index_block = beat_context.blocks.input_index_blocks[1 + bar_context.current_table_b_index_block_index];
                const index_schema = schema.TableIndex.from(index_block);
                assert(previous_schema == null or stdx.equal_bytes(schema.TableIndex, &previous_schema.?, &index_schema));
                previous_schema = index_schema;

                const data_blocks_used = index_schema.data_blocks_used(index_block);
                assert(bar_context.current_table_b_value_block_index < data_blocks_used);

                const data_block_addresses = index_schema.data_addresses_used(index_block);
                const data_block_checksums = index_schema.data_checksums_used(index_block);

                // Issue as many reads as we have space for.
                var data_blocks_read: usize = 0;

                // Try read in as many value blocks as this index block has...
                while (data_blocks_read < data_blocks_used) {
                    beat_context.grid_reads[read_target].target = compaction;
                    beat_context.grid_reads[read_target].hack = read_target;
                    compaction.grid.read_block(
                        .{ .from_local_or_global_storage = blip_read_data_callback },
                        &beat_context.grid_reads[read_target].read,
                        data_block_addresses[data_blocks_read],
                        data_block_checksums[data_blocks_read].value,
                        .{ .cache_read = true, .cache_write = true },
                    );

                    read.pending_data_reads += 1;
                    read_target += 1;
                    data_blocks_read += 1;

                    // But, once our read buffer is full, break out of the outer loop.
                    if (data_blocks_read < beat_context.blocks.input_data_blocks[beat_context.read_current_split][1].len) {
                        break :outer;
                    }
                }
            }

            // Either we have pending data reads, in which case blip_read_next_tick gets called by
            // blip_read_data_callback once all reads are done, or we don't, in which case call it
            // here via next_tick.
            if (read.pending_data_reads == 0) {
                compaction.grid.on_next_tick(blip_read_next_tick, &read.next_tick);
            }
        }

        fn blip_read_data_callback(grid_read: *Grid.Read, block: BlockPtrConst) void {
            const compaction: *Compaction = @alignCast(@ptrCast(@fieldParentPtr(Grid.FatRead, "read", grid_read).target));
            assert(compaction.bar_context != null);
            assert(compaction.beat_context != null);

            const bar_context = &compaction.bar_context.?;
            _ = bar_context;
            const beat_context = &compaction.beat_context.?;
            _ = block;

            assert(beat_context.read != null);
            const read = &beat_context.read.?;

            read.pending_data_reads -= 1;
            if (read.pending_data_reads != 0) return;

            const d = read.timer.read();
            std.log.info("Took {} to read all blocks - {}", .{ std.fmt.fmtDuration(d), read.timer_read });

            // Call the next tick handler directly. We have already been next_tick'd through, so this is safe from stack
            // overflows.
            blip_read_next_tick(&read.next_tick);
        }

        fn blip_read_next_tick(next_tick: *Grid.NextTick) void {
            const read_context: *?BeatContext.ReadContext = @ptrCast(@fieldParentPtr(BeatContext.ReadContext, "next_tick", next_tick));
            const beat_context = @fieldParentPtr(BeatContext, "read", read_context);
            // const beat_context: *?BeatContext = @ptrCast(@fieldParentPtr(BeatContext, "pipeline_context", pipeline_context));
            // const compaction = @fieldParentPtr(Compaction, "beat_context", beat_context);
            // _ = compaction;

            beat_context.deactivate_and_assert_and_callback(.read, null, null);
        }

        fn calculate_values_in(compaction: *Compaction) ValuesIn {
            assert(compaction.beat_context != null);
            const bar_context = &compaction.bar_context.?;
            const beat_context = &compaction.beat_context.?;
            const cpu = &beat_context.merge.?;

            const blocks_a = beat_context.blocks.input_data_blocks[beat_context.merge_current_split][0];
            const blocks_b = beat_context.blocks.input_data_blocks[beat_context.merge_current_split][1];

            // FIXME: TODO handle when we have b tables lol
            const values_b = Table.data_block_values_used(blocks_b[cpu.current_block_b]);

            // // Assert that we're reading data blocks in key order.
            // const values_in = compaction.values_in[index];
            // assert(values_in.len > 0);
            // if (constants.verify) {
            //     for (values_in[0 .. values_in.len - 1], values_in[1..]) |*value, *value_next| {
            //         assert(key_from_value(value) < key_from_value(value_next));
            //     }
            // }
            // const first_key = key_from_value(&values_in[0]);
            // const last_key = key_from_value(&values_in[values_in.len - 1]);
            // if (compaction.last_keys_in[index]) |last_key_prev| {
            //     assert(last_key_prev < first_key);
            // }
            // if (values_in.len > 1) {
            //     assert(first_key < last_key);
            // }
            // compaction.last_keys_in[index] = last_key;

            // FIXME: Assert we're not exhausted.
            switch (compaction.bar_context.?.table_info_a) {
                .immutable => {
                    return .{
                        UnifiedIterator{ .dispatcher = .{
                            .table_memory_iterator = Tree.TableMemory.Iterator.init(bar_context.table_info_a.immutable, bar_context.current_block_a_idx),
                        } },
                        UnifiedIterator{ .dispatcher = .{
                            .value_block_iterator = ValueBlockIterator.init(values_b, bar_context.current_block_b_idx),
                        } },
                    };
                },
                .disk => {
                    assert(false); // FIXME: Unimplemented for now.
                    const values_a = Table.data_block_values_used(blocks_a[cpu.current_block_a]);
                    return .{
                        UnifiedIterator{ .dispatcher = .{
                            .value_block_iterator = ValueBlockIterator.init(values_a, bar_context.current_block_a_idx),
                        } },
                        UnifiedIterator{ .dispatcher = .{
                            .value_block_iterator = ValueBlockIterator.init(values_b, bar_context.current_block_b_idx),
                        } },
                    };
                },
            }
        }

        /// Perform CPU merge work, to transform our input tables to our output tables.
        /// This has a data dependency on both the read buffers and the write buffers for the current
        /// split.
        ///
        // A blip is over when one of the following condition are met, considering we don't want to output partial data blocks unless we really really have to:
        // * We have no more input remaining, at all - the bar is done. This will likely result in a partially full data block.
        // * We have no more output data blocks remaining - we might need more blips, but that's up to the forest to orchestrate.
        // * We have no more output index blocks remaining - we might have a partial data block here, but that's OK.
        // * We have no more input blocks remaining:
        //   * FIXME: Flesh out the cases here
        pub fn blip_merge(compaction: *Compaction, callback: BlipCallback, ptr: *anyopaque) void {
            assert(compaction.bar_context != null);
            assert(compaction.beat_context != null);

            const bar_context = &compaction.bar_context.?;
            const beat_context = &compaction.beat_context.?;

            beat_context.activate_and_assert(.merge, callback, ptr);

            assert(beat_context.merge != null);
            const merge = &beat_context.merge.?;

            var beat_exhausted = false;
            var bar_exhausted = false;

            // FIXME: No while true loops!
            var i: usize = 0;
            outer: while (true) : (i += 1) {
                assert(bar_context.table_builder.value_count < Table.layout.block_value_count_max);

                var values_in = compaction.calculate_values_in();

                var values_in_a_len = values_in[0].remaining();
                var values_in_b_len = values_in[1].remaining();

                // Set the index block if needed.
                if (bar_context.table_builder.state == .no_blocks) {
                    bar_context.table_builder.set_index_block(bar_context.output_index_blocks[bar_context.current_index_block]);
                }

                // Set the data block if needed.
                if (bar_context.table_builder.state == .index_block) {
                    bar_context.table_builder.set_data_block(beat_context.blocks.output_data_blocks[beat_context.merge_current_split][merge.current_output_data_block]);
                } else {
                    // FIXME: Figure this out.
                    assert(i > 0);
                }

                std.log.info("blip_merge: values_a: {} and values_b: {} current cpu split: {}", .{ values_in_a_len, values_in_b_len, beat_context.merge_current_split });

                // We are responsible for not iterating again if all work is done.
                assert(values_in_a_len > 0 or values_in_b_len > 0);

                // Loop through the CPU work until we have nothing left.
                while (values_in_a_len > 0 or values_in_b_len > 0) {
                    if (values_in_a_len == 0) {
                        values_in[1].copy(&bar_context.table_builder);
                    } else if (values_in_b_len == 0) {
                        if (bar_context.drop_tombstones) {
                            copy_drop_tombstones(&bar_context.table_builder, &values_in);
                        } else {
                            values_in[0].copy(&bar_context.table_builder);
                        }
                    } else {
                        merge_fn(&bar_context.table_builder, &values_in, bar_context.drop_tombstones);
                    }

                    // FIXME: Update these in a nicer way...
                    const values_in_a_len_new = values_in[0].remaining();
                    const values_in_b_len_new = values_in[1].remaining();

                    // Save our iterator position
                    bar_context.current_block_a_idx = values_in[0].get_source_index();
                    bar_context.current_block_b_idx = values_in[1].get_source_index();

                    beat_context.input_values_processed += (values_in_a_len - values_in_a_len_new) + (values_in_b_len - values_in_b_len_new);
                    values_in_a_len = values_in_a_len_new;
                    values_in_b_len = values_in_b_len_new;

                    // When checking if we're done, there are two things we need to consider:
                    // 1. Have we finished our input entirely? If so, we flush what we have - it'll be a partial block but that's OK.
                    // 2. Have we reached our value_count_goal? If so, we'll flush at the next complete data block.
                    //
                    // This means that we'll potentially overrun our value_count_goal by up to a full data block.
                    // FIXME: Temp for now
                    bar_exhausted = values_in_a_len + values_in_b_len == 0;
                    beat_exhausted = beat_context.input_values_processed > bar_context.per_beat_input_goal;

                    std.log.info("We've done a total of: {} this beat VS target of {} BarE {} BeatE {}", .{ beat_context.input_values_processed, bar_context.per_beat_input_goal, bar_exhausted, beat_exhausted });

                    switch (compaction.check_and_finish_blocks(.{ .index_block = bar_exhausted, .data_block = bar_exhausted })) {
                        .unfinished_data_block => continue,

                        .finished_data_block => {
                            if (beat_exhausted) {
                                break :outer;
                            }
                        },

                        // If we have to write, we need to break out of the outer loop.
                        .need_write => {
                            // FIXME: Gross - maybe make transitioning between needed pipeline states an explicit fn
                            // FIXME: XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
                            // FIXME: SUPER NB! We need to tell our write stage how many blocks to write! This is beat context since it crosses stages!
                            // beat_context.write.data_blocks_count = merge.current_output_data_block;
                            break :outer;
                        },
                    }
                }
            }

            // FIXME: Check at least one output value.
            // assert(filled <= target.len);
            // if (filled == 0) assert(Table.usage == .secondary_index);

            const d = merge.timer.read();
            std.log.info("Took {} to CPU merge block", .{std.fmt.fmtDuration(d)});

            beat_context.deactivate_and_assert_and_callback(.merge, beat_exhausted or bar_exhausted, bar_exhausted);
        }

        // FIXME: Make input_exhausted a struct rather
        fn check_and_finish_blocks(compaction: *Compaction, force_flush: struct { index_block: bool, data_block: bool }) enum { unfinished_data_block, finished_data_block, need_write } {
            assert(compaction.bar_context != null);
            assert(compaction.beat_context != null);

            const bar_context = &compaction.bar_context.?;
            const beat_context = &compaction.beat_context.?;

            assert(beat_context.merge != null);
            const merge = &beat_context.merge.?;

            // If the index block is force flushed, the data block must be too.
            assert(!force_flush.index_block or force_flush.data_block);

            const table_builder = &bar_context.table_builder;

            // FIXME: For now we don't distinguish between needing index / data writes; we just write whatever
            // we can, based on the _completed_count fields.
            var need_write = false;
            var finished_data_block = false;

            // Flush the data block if needed.
            if (table_builder.data_block_full() or
                table_builder.index_block_full() or
                (force_flush.data_block and !table_builder.data_block_empty()))
            {
                table_builder.data_block_finish(.{
                    .cluster = compaction.grid.superblock.working.cluster,
                    .address = compaction.grid.acquire(compaction.beat_context.?.grid_reservation.?),
                    .snapshot_min = snapshot_min_for_table_output(bar_context.op_min),
                    .tree_id = compaction.tree_config.id,
                });
                std.log.info("Finished data block", .{});

                merge.current_output_data_block += 1;
                finished_data_block = true;
                if (merge.current_output_data_block == beat_context.blocks.output_data_blocks[beat_context.merge_current_split].len or force_flush.data_block) {
                    need_write = true;
                }
            }

            // Flush the index block if needed.
            if (table_builder.index_block_full() or
                // If the input is exhausted then we need to flush all blocks before finishing.
                (force_flush.index_block and !table_builder.index_block_empty()))
            {
                const table = table_builder.index_block_finish(.{
                    .cluster = compaction.grid.superblock.working.cluster,
                    .address = compaction.grid.acquire(compaction.beat_context.?.grid_reservation.?),
                    .snapshot_min = snapshot_min_for_table_output(bar_context.op_min),
                    .tree_id = compaction.tree_config.id,
                });
                std.log.info("Finished index block", .{});

                // FIXME: Write pipelining and the bar global index block situation.
                bar_context.current_index_block += 1;
                if (bar_context.current_index_block == bar_context.output_index_blocks.len or force_flush.index_block) {
                    need_write = true;
                }

                // Make this table visible at the end of this bar.
                bar_context.manifest_entries.append_assume_capacity(.{
                    .operation = .insert_to_level_b,
                    .table = table,
                });
            }

            if (need_write) return .need_write;
            if (finished_data_block) return .finished_data_block;

            return .unfinished_data_block;
        }

        /// Perform write IO to write our output_index_blocks and output_data_blocks to disk.
        pub fn blip_write(compaction: *Compaction, callback: BlipCallback, ptr: *anyopaque) void {
            // FIXME: Is there a point to asserting != null if we use .? later?
            assert(compaction.bar_context != null);
            assert(compaction.beat_context != null);

            const bar_context = &compaction.bar_context.?;
            const beat_context = &compaction.beat_context.?;

            assert(beat_context.write != null);
            const write = &beat_context.write.?;

            beat_context.activate_and_assert(.write, callback, ptr);

            std.log.info("Current write split: {}", .{beat_context.write_current_split});

            var write_index: usize = 0;

            // Write any complete index blocks.
            for (bar_context.output_index_blocks[0..bar_context.current_index_block]) |*block| {
                // std.log.info("Issuing a write for split {} block {}", .{ write_current_split, i });

                beat_context.grid_writes[write_index].target = compaction;
                write.pending_writes += 1;
                compaction.grid.create_block(blip_write_callback, &beat_context.grid_writes[write_index].write, block);
                write_index += 1;
            }

            // Write any complete data blocks.
            for (beat_context.blocks.output_data_blocks[beat_context.write_current_split][0..write.data_blocks_count]) |*block| {
                // std.log.info("Issuing a write for split {} block {}", .{ write_current_split, i });

                beat_context.grid_writes[write_index].target = compaction;
                write.pending_writes += 1;
                compaction.grid.create_block(blip_write_callback, &beat_context.grid_writes[write_index].write, block);
                write_index += 1;
            }

            const d = write.timer.read();
            std.log.info("Took {} to create {} blocks", .{ std.fmt.fmtDuration(d), write_index });
            write.timer.reset();

            // FIXME: From 2023-12-21
            // FIXME: Pace our compaction by input *values* not input blocks. Blocks might be empty, values will be a far better metric.
            // FIXME: Whenever we run and pace compaction, in the one worst case we'll have 9 input tables forming 7 output tables, and the
            // other will be 9 input tables forming 9 output tables. We should assert that we always do this.
            // The other note is that we don't want to hang on to data blocks across beat boundaries, so we'll always end when one is full
            // and not try to be too perfect.
            // FIXME: The big idea is to make compaction pacing explicit and asserted behaviour rather than just an implicit property of the code

            // FIXME: We need to deactivate if we have 0 blocks to write, and call our callback etc.
            // beat_context.deactivate_and_assert(.write);
        }

        fn blip_write_callback(grid_write: *Grid.Write) void {
            const compaction: *Compaction = @alignCast(@ptrCast(@fieldParentPtr(Grid.FatWrite, "write", grid_write).target));
            assert(compaction.bar_context != null);
            assert(compaction.beat_context != null);

            const bar_context = &compaction.bar_context.?;
            _ = bar_context;
            const beat_context = &compaction.beat_context.?;

            assert(beat_context.write != null);
            const write = &beat_context.write.?;
            write.pending_writes -= 1;

            // std.log.info("blip_write_callback for split {}", .{write_current_split});

            if (write.pending_writes == 0) {
                const duration = write.timer.read();
                std.log.info("all writes done - took {}! callback time!!", .{std.fmt.fmtDuration(duration)});
                beat_context.deactivate_and_assert_and_callback(.write, null, null);
            }
        }

        pub fn beat_grid_forfeit(compaction: *Compaction) void {
            // FIXME: Big hack - see beat_end comment
            if (compaction.beat_context == null) {
                return;
            }

            assert(compaction.bar_context != null);
            assert(compaction.beat_context != null);

            const beat_context = &compaction.beat_context.?;

            beat_context.assert_all_inactive();
            assert(compaction.bar_context.?.table_builder.data_block_empty());
            // assert(compaction.bar_context.?.table_builder.state == .index_block); // Hmmm

            if (beat_context.grid_reservation) |grid_reservation| {
                std.log.info("forfeiting... {}", .{grid_reservation});
                compaction.grid.forfeit(grid_reservation);
                // We set the whole beat_context to null later.
            } else {
                assert(compaction.bar_context.?.move_table);
            }

            // FIXME: Hack, real compaction will need to set this obvs
            // std.log.info("WTF: {}", .{compaction.bar_context.?.tree.table_immutable.mutability.immutable.flushed});
            compaction.bar_context.?.tree.table_immutable.mutability.immutable.flushed = true;

            // Our beat is done!
            compaction.beat_context = null;
        }

        /// FIXME: Describe
        pub fn bar_finish(compaction: *Compaction, op: u64, tree: *Tree) void {
            std.log.info("bar_finish running for compaction: {s} into level: {}", .{ compaction.tree_config.name, compaction.level_b });

            // If we're the compaction for immutable -> level 0, we need to swap our mutable / immutable
            // tables too. This needs to happen at the end of the first ever bar, which would normally
            // not have any work to do, so put it before our asserts.
            // FIXME: Do this in a better way
            if (compaction.level_b == 0) {
                // FIXME: Figure out wtf I'm doing with snapshots
                tree.swap_mutable_and_immutable(
                    snapshot_min_for_table_output(op + 1),
                );
            }

            if (compaction.bar_context == null and op + 1 == constants.lsm_batch_multiple) {
                return;
            }

            // Fixme: hmm
            if (compaction.bar_context == null) {
                return;
            }

            assert(compaction.beat_context == null);
            assert(compaction.bar_context != null);

            const bar_context = &compaction.bar_context.?;

            // BLAAAH deal with this better!
            if (compaction.bar_context == null) {
                return;
            }

            // FIXME: Assert our input has been fully exhausted.

            // Each compaction's manifest updates are deferred to the end of the last
            // bar to ensure:
            // - manifest log updates are ordered deterministically relative to one another, and
            // - manifest updates are not visible until after the blocks are all on disk.
            const manifest = &bar_context.tree.manifest;
            const level_b = compaction.level_b;
            const snapshot_max = snapshot_max_for_table_input(bar_context.op_min);

            if (bar_context.move_table) {
                // If no compaction is required, don't update snapshot_max.
            } else {
                // These updates MUST precede insert_table() and move_table() since they use
                // references to modify the ManifestLevel in-place.
                switch (bar_context.table_info_a) {
                    .immutable => {},
                    .disk => |table_info| {
                        manifest.update_table(level_b - 1, snapshot_max, table_info);
                    },
                }
                for (bar_context.range_b.tables.const_slice()) |table| {
                    manifest.update_table(level_b, snapshot_max, table);
                }
            }

            for (bar_context.manifest_entries.slice()) |*entry| {
                switch (entry.operation) {
                    .insert_to_level_b => manifest.insert_table(level_b, &entry.table),
                    .move_to_level_b => manifest.move_table(level_b - 1, level_b, &entry.table),
                }
            }

            // Our bar is done!
            compaction.bar_context = null;
        }

        /// If we can just move the table, don't bother with merging.
        fn move_table(compaction: *Compaction) void {
            const bar_context = &compaction.bar_context.?;

            log.info(
                "{s}: Moving table: level_b={}",
                .{ compaction.tree_config.name, compaction.level_b },
            );

            const snapshot_max = snapshot_max_for_table_input(bar_context.op_min);
            const table_a = bar_context.table_info_a.disk.table_info;
            assert(table_a.snapshot_max >= snapshot_max);

            bar_context.manifest_entries.append_assume_capacity(.{
                .operation = .move_to_level_b,
                .table = table_a.*,
            });
        }

        // TODO: Support for LSM snapshots would require us to only remove blocks
        // that are invisible.
        fn release_table_blocks(compaction: *Compaction, index_block: BlockPtrConst) void {
            // Release the table's block addresses in the Grid as it will be made invisible.
            // This is safe; compaction.index_block_b holds a copy of the index block for a
            // table in Level B. Additionally, compaction.index_block_a holds
            // a copy of the index block for the Level A table being compacted.

            const grid = compaction.grid;
            const index_schema = schema.TableIndex.from(index_block);
            for (index_schema.data_addresses_used(index_block)) |address| grid.release(address);
            grid.release(Table.block_address(index_block));
        }

        ////////////// The actual CPU merging methods below. //////////////
        // TODO: Add benchmarks for all of these specifically.
        /// Copy values from table_a to table_b, dropping tombstones as we go.
        fn copy_drop_tombstones(table_builder: *Table.Builder, values_in: *ValuesIn) void {
            std.log.info("Entering copy_drop_tombstones...", .{});
            assert(values_in[1].remaining() == 0);
            assert(table_builder.value_count < Table.layout.block_value_count_max);

            // Copy variables locally to ensure a tight loop.
            var values_in_a = values_in[0];
            const values_out = table_builder.data_block_values();
            var values_out_index = table_builder.value_count;

            // Merge as many values as possible.
            while (values_in_a.next()) |value_a| {
                if (tombstone(&value_a)) {
                    // TODO: What's the impact of this check?
                    assert(Table.usage != .secondary_index);
                    continue;
                }
                values_out[values_out_index] = value_a;
                values_out_index += 1;

                if (values_out_index == values_out.len) break;
            }

            // Copy variables back out.
            values_in[0] = values_in_a;
            table_builder.value_count = values_out_index;

            std.log.info("Set table builder value to: {}", .{table_builder.value_count});
        }

        fn merge_fn(table_builder: *Table.Builder, values_in: *ValuesIn, drop_tombstones: bool) void {
            _ = drop_tombstones;
            _ = values_in;
            _ = table_builder;
            assert(false); // Merge todo still
            //     assert(values_in[0].remaining() > 0);
            //     assert(values_in[1].remaining() > 0);
            //     assert(table_builder.value_count < Table.layout.block_value_count_max);

            //     // Copy variables locally to ensure a tight loop.
            //     const values_in_a = values_in[0];
            //     const values_in_b = values_in[1];
            //     const values_out = table_builder.data_block_values();
            //     var values_in_a_index: usize = 0;
            //     var values_in_b_index: usize = 0;
            //     var values_out_index = table_builder.value_count;

            //     // Merge as many values as possible.
            //     while (values_in_a_index < values_in_a.len and
            //         values_in_b_index < values_in_b.len and
            //         values_out_index < values_out.len)
            //     {
            //         const value_a = &values_in_a[values_in_a_index];
            //         const value_b = &values_in_b[values_in_b_index];
            //         switch (std.math.order(key_from_value(value_a), key_from_value(value_b))) {
            //             .lt => {
            //                 values_in_a_index += 1;
            //                 if (drop_tombstones and
            //                     tombstone(value_a))
            //                 {
            //                     assert(Table.usage != .secondary_index);
            //                     continue;
            //                 }
            //                 values_out[values_out_index] = value_a.*;
            //                 values_out_index += 1;
            //             },
            //             .gt => {
            //                 values_in_b_index += 1;
            //                 values_out[values_out_index] = value_b.*;
            //                 values_out_index += 1;
            //             },
            //             .eq => {
            //                 values_in_a_index += 1;
            //                 values_in_b_index += 1;

            //                 if (Table.usage == .secondary_index) {
            //                     // Secondary index optimization --- cancel out put and remove.
            //                     assert(tombstone(value_a) != tombstone(value_b));
            //                     continue;
            //                 } else if (drop_tombstones) {
            //                     if (tombstone(value_a)) {
            //                         continue;
            //                     }
            //                 }

            //                 values_out[values_out_index] = value_a.*;
            //                 values_out_index += 1;
            //             },
            //         }
            //     }

            //     // Copy variables back out.
            //     values_in[0] = values_in_a[values_in_a_index..];
            //     values_in[1] = values_in_b[values_in_b_index..];
            //     table_builder.value_count = values_out_index;
        }
    };
}

fn snapshot_max_for_table_input(op_min: u64) u64 {
    return snapshot_min_for_table_output(op_min) - 1;
}

pub fn snapshot_min_for_table_output(op_min: u64) u64 {
    assert(op_min > 0);
    assert(op_min % @divExact(constants.lsm_batch_multiple, 2) == 0);
    return op_min + @divExact(constants.lsm_batch_multiple, 2);
}

/// Returns the first op of the compaction (Compaction.op_min) for a given op/beat.
///
/// After this compaction finishes:
/// - `op_min + half_bar_beat_count - 1` will be the input tables' snapshot_max.
/// - `op_min + half_bar_beat_count` will be the output tables' snapshot_min.
///
/// Each half-bar has a separate op_min (for deriving the output snapshot_min) instead of each full
/// bar because this allows the output tables of the first half-bar's compaction to be prefetched
/// against earlier — hopefully while they are still warm in the cache from being written.
///
///
/// These charts depict the commit/compact ops over a series of
/// commits and compactions (with lsm_batch_multiple=8).
///
/// Legend:
///
///   ┼   full bar (first half-bar start)
///   ┬   half bar (second half-bar start)
///       This is incremented at the end of each compact().
///   .   op is in mutable table (in memory)
///   ,   op is in immutable table (in memory)
///   #   op is on disk
///   ✓   checkpoint() may follow compact()
///
///   0 2 4 6 8 0 2 4 6
///   ┼───┬───┼───┬───┼
///   .       ╷       ╷     init(superblock.commit_min=0)⎤ Compaction is effectively a noop for the
///   ..      ╷       ╷     commit;compact( 1) start/end ⎥ first bar because there are no tables on
///   ...     ╷       ╷     commit;compact( 2) start/end ⎥ disk yet, and no immutable table to
///   ....    ╷       ╷     commit;compact( 3) start/end ⎥ flush.
///   .....   ╷       ╷     commit;compact( 4) start/end ⎥
///   ......  ╷       ╷     commit;compact( 5) start/end ⎥ This applies:
///   ....... ╷       ╷     commit;compact( 6) start/end ⎥ - when the LSM is starting on a freshly
///   ........╷       ╷     commit;compact( 7) start    ⎤⎥   formatted data file, and also
///   ,,,,,,,,.       ╷  ✓         compact( 7)       end⎦⎦ - when the LSM is recovering from a crash
///   ,,,,,,,,.       ╷     commit;compact( 8) start/end     (see below).
///   ,,,,,,,,..      ╷     commit;compact( 9) start/end
///   ,,,,,,,,...     ╷     commit;compact(10) start/end
///   ,,,,,,,,....    ╷     commit;compact(11) start/end
///   ,,,,,,,,.....   ╷     commit;compact(12) start/end
///   ,,,,,,,,......  ╷     commit;compact(13) start/end
///   ,,,,,,,,....... ╷     commit;compact(14) start/end
///   ,,,,,,,,........╷     commit;compact(15) start    ⎤
///   ########,,,,,,,,╷  ✓         compact(15)       end⎦
///   ########,,,,,,,,.     commit;compact(16) start/end
///   ┼───┬───┼───┬───┼
///   0 2 4 6 8 0 2 4 6
///   ┼───┬───┼───┬───┼                                    Recover with a checkpoint taken at op 15.
///   ########        ╷     init(superblock.commit_min=7)  At op 15, ops 8…15 are in memory, so they
///   ########.       ╷     commit        ( 8) start/end ⎤ were dropped by the crash.
///   ########..      ╷     commit        ( 9) start/end ⎥
///   ########...     ╷     commit        (10) start/end ⎥ But compaction is not run for ops 8…15
///   ########....    ╷     commit        (11) start/end ⎥ because it was already performed
///   ########.....   ╷     commit        (12) start/end ⎥ before the checkpoint.
///   ########......  ╷     commit        (13) start/end ⎥
///   ########....... ╷     commit        (14) start/end ⎥ We can begin to compact again at op 16,
///   ########........╷     commit        (15) start    ⎤⎥ because those compactions (if previously
///   ########,,,,,,,,╷  ✓                (15)       end⎦⎦ performed) are not included in the
///   ########,,,,,,,,.     commit;compact(16) start/end   checkpoint.
///   ┼───┬───┼───┬───┼
///   0 2 4 6 8 0 2 4 6
///
/// Notice how in the checkpoint recovery example above, we are careful not to `compact(op)` twice
/// for any op (even if we crash/recover), since that could lead to differences between replicas'
/// storage. The last bar of `commit()`s is always only in memory, so it is safe to repeat.
pub fn compaction_op_min(op: u64) u64 {
    _ = op;
    // return op - op % half_bar_beat_count;
}
