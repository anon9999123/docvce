from typing import Any, Dict, List, Optional, Union

from atria.core.data.data_transforms import DataTransform


class DoclayNetLabelTransform(DataTransform):
    def __init__(self, labels: List[str], key: Optional[Union[str, List[str]]] = None):
        super().__init__(key)
        self.labels = labels

    def _apply_transform(self, label: Dict[str, Any]) -> Dict[str, Any]:
        return self.labels.index(label)

    def __repr__(self):
        return f"{self.__class__.__name__}(labels={self.labels})"
