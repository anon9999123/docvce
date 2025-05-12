import gzip
from typing import Any, Dict, List, Optional, Union

import pytesseract
from atria.core.constants import DataKeys
from atria.core.data.data_transforms import DataTransform


class HocrExtractTransform(DataTransform):
    def __init__(
        self,
        lang: Optional[str] = None,
        config: str = "--psm 1 --oem 1",
        extension="hocr",
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key)
        self.lang = lang
        self.config = config
        self.extension = extension

    def _apply_transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        hocr = pytesseract.image_to_pdf_or_hocr(
            sample[DataKeys.IMAGE],
            lang=self.lang,
            config=self.config,
            extension=self.extension,
        )
        sample[DataKeys.HOCR] = gzip.compress(hocr)
        return sample

    def __repr__(self):
        return f"{self.__class__.__name__}(lang={self.lang}, config={self.config})"
