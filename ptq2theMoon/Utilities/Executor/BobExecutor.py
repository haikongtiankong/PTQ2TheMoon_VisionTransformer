from ptq2theMoon import *
from ptq2theMoon.Utilities.Executor.myHook import MyHook


class BobbyExecutor(TorchExecutor):
    def __init__(self, pickle_path: str, type: int) -> None:
        super().__init__(pickle_path, type)
        self.pickle_path = pickle_path
        self.type = type

    @torch.no_grad()
    @empty_ppq_cache
    def tracing_operation_meta(
            self,
            inputs: Union[dict, list, torch.Tensor],
            output_names: List[str] = None,
    ) -> None:
        hooks = {}
        for op_name, operation in self._graph.operations.items():
            hooks[op_name] = MyHook(operation=operation, pickle_path=self.pickle_path, type=self.type)

        self.__forward(
            inputs=inputs,
            output_names=output_names,
            executing_order=self._executing_order,
            hooks=hooks)