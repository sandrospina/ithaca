//! The Web Worker wrapper around [`SimCore`].

use gloo_worker::{HandlerId, Worker, WorkerScope};

use crate::protocol::{SimRequest, SimResponse};
use crate::sim::SimCore;

pub struct SimWorker {
    core: SimCore,
}

impl Worker for SimWorker {
    type Message = ();
    type Input = SimRequest;
    type Output = SimResponse;

    fn create(_scope: &WorkerScope<Self>) -> Self {
        Self { core: SimCore::default() }
    }

    fn update(&mut self, _scope: &WorkerScope<Self>, _msg: Self::Message) {}

    fn received(&mut self, scope: &WorkerScope<Self>, msg: Self::Input, id: HandlerId) {
        scope.respond(id, self.core.handle(msg));
    }
}
