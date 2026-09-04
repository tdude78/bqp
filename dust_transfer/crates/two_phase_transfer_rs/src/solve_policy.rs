const TRANSFER_PAIR_PAR_THRESHOLD: usize = 4;

pub fn should_parallelize_selected_pairs(selected_pairs: usize) -> bool {
    should_parallelize_selected_pairs_for_policy(
        selected_pairs,
        rayon::current_thread_index().is_some(),
        rayon::current_num_threads(),
    )
}

const fn should_parallelize_selected_pairs_for_policy(
    selected_pairs: usize,
    nested: bool,
    thread_count: usize,
) -> bool {
    !nested && thread_count > 1 && selected_pairs >= TRANSFER_PAIR_PAR_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_pair_policy_rejects_pairs_below_threshold() {
        assert!(!should_parallelize_selected_pairs_for_policy(
            TRANSFER_PAIR_PAR_THRESHOLD - 1,
            false,
            8,
        ));
    }

    #[test]
    fn selected_pair_policy_rejects_nested_worker() {
        assert!(!should_parallelize_selected_pairs_for_policy(
            TRANSFER_PAIR_PAR_THRESHOLD,
            true,
            8,
        ));
    }

    #[test]
    fn selected_pair_policy_rejects_single_thread_pool() {
        assert!(!should_parallelize_selected_pairs_for_policy(
            TRANSFER_PAIR_PAR_THRESHOLD,
            false,
            1,
        ));
    }

    #[test]
    fn selected_pair_policy_allows_outer_multi_thread_pool() {
        assert!(should_parallelize_selected_pairs_for_policy(
            TRANSFER_PAIR_PAR_THRESHOLD,
            false,
            8,
        ));
    }
}
