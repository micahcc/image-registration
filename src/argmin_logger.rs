use anyhow::Result;
use argmin::core::observers::Observe;
use argmin::core::State;

pub struct ArgminLogger {}

impl<I> Observe<I> for ArgminLogger
where
    I: State,
{
    /// Log basic information about the optimization after initialization.
    fn observe_init(
        &mut self,
        msg: &str,
        state: &I,
        _kv: &argmin::core::KV,
    ) -> Result<(), argmin::core::Error> {
        log::info!(
            "{msg} best_cost={}, cost={}, iter={}",
            state.get_best_cost(),
            state.get_cost(),
            state.get_iter(),
        );
        Ok(())
    }

    /// Logs information about the progress of the optimization after every iteration.
    fn observe_iter(
        &mut self,
        state: &I,
        _kv: &argmin::core::KV,
    ) -> Result<(), argmin::core::Error> {
        log::info!(
            "best_cost={}, cost={}, iter={}",
            state.get_best_cost(),
            state.get_cost(),
            state.get_iter(),
        );
        Ok(())
    }
}
