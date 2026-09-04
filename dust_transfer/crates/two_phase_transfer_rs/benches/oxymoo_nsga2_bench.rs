use anyhow::{bail, Result};
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use two_phase_transfer_rs::{
    crowding_distance, fast_nondominated_sort, Nsga2, Nsga2Config, Problem, SortConfig,
    VariableKind, VariableSpec,
};

fn require_bench_success<T>(result: &Result<T>, operation: &'static str) {
    if let Err(error) = result {
        assert!(
            result.is_ok(),
            "BENCH-HARNESS: {operation} failed: {error:#}"
        );
    }
}

#[derive(Clone)]
struct ZdtLikeProblem {
    variables: Vec<VariableSpec>,
}

impl ZdtLikeProblem {
    fn new(n_variables: usize) -> Self {
        Self {
            variables: vec![VariableSpec::new(0.0, 1.0, VariableKind::Continuous); n_variables],
        }
    }
}

impl Problem for ZdtLikeProblem {
    fn variable_specs(&self) -> &[VariableSpec] {
        &self.variables
    }

    fn objective_count(&self) -> usize {
        2
    }

    fn evaluate(&self, decision: &[f64], objectives: &mut [f64]) -> Result<f64> {
        let Some((&f1, tail)) = decision.split_first() else {
            bail!("ZDT-like benchmark needs at least two decision variables");
        };
        let tail_count = u32::try_from(tail.len()).map_err(|_| {
            anyhow::anyhow!("ZDT-like benchmark tail length exceeds u32: {}", tail.len())
        })?;
        if tail_count == 0 {
            bail!("ZDT-like benchmark needs at least two decision variables");
        }
        let tail_sum: f64 = tail.iter().copied().sum();
        let g = 1.0 + 9.0 * tail_sum / f64::from(tail_count);
        let h = 1.0 - (f1 / g).sqrt();
        let Some((first_objective, remaining_objectives)) = objectives.split_first_mut() else {
            bail!("ZDT-like benchmark needs two objective slots");
        };
        let Some((second_objective, _)) = remaining_objectives.split_first_mut() else {
            bail!("ZDT-like benchmark needs two objective slots");
        };
        *first_objective = f1;
        *second_objective = g * h;
        Ok(0.0)
    }
}

fn make_objectives_256x2() -> (Vec<f64>, Vec<f64>) {
    let mut objectives = Vec::with_capacity(512);
    let mut cv = Vec::with_capacity(256);
    for i in 0_u32..256 {
        let t = f64::from(i) / 256.0;
        objectives.push(t);
        objectives.push(1.0 - t.sqrt());
        cv.push(0.0);
    }
    (objectives, cv)
}

fn bench_sort_and_crowding(c: &mut Criterion) {
    let (objectives, cv) = make_objectives_256x2();
    c.bench_function("fast_nondominated_sort_256x2", |b| {
        b.iter(|| {
            let result = fast_nondominated_sort(
                black_box(&objectives),
                256,
                2,
                black_box(&cv),
                SortConfig::default(),
            );
            require_bench_success(&result, "fast_nondominated_sort");
            if let Ok(fronts) = result {
                black_box(fronts);
            }
        });
    });

    let front: Vec<usize> = (0..256).collect();
    c.bench_function("crowding_distance_256x2", |b| {
        b.iter(|| {
            let result = crowding_distance(black_box(&objectives), 256, 2, black_box(&front));
            require_bench_success(&result, "crowding_distance");
            if let Ok(distances) = result {
                black_box(distances);
            }
        });
    });
}

fn bench_nsga2_generation(c: &mut Criterion) {
    let config = Nsga2Config {
        population_size: 128,
        generations: 1,
        seed: 42,
        ..Nsga2Config::default()
    };
    c.bench_function("nsga2_one_generation_zdt_like", |b| {
        b.iter(|| {
            let optimizer = Nsga2::new(ZdtLikeProblem::new(16), config.clone());
            require_bench_success(&optimizer, "Nsga2::new");
            if let Ok(mut optimizer) = optimizer {
                let result = optimizer.run();
                require_bench_success(&result, "Nsga2::run");
                if let Ok(result) = result {
                    black_box(result);
                }
            }
        });
    });
}

criterion_group!(benches, bench_sort_and_crowding, bench_nsga2_generation,);
criterion_main!(benches);
