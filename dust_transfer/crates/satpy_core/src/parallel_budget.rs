//! Hardware core-count helper.

#[inline]
#[must_use]
pub fn available_cores() -> usize {
    std::thread::available_parallelism().map_or(1, |parallelism| parallelism.get().max(1))
}
