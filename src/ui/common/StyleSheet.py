# coding: utf-8
from enum import Enum

from qfluentwidgets import (StyleSheetBase,
                            Theme,
                            qconfig,)


class StyleSheet(StyleSheetBase, Enum):
    """ Style sheet  """

    LINK_CARD = "link_card"
    SAMPLE_CARD = "sample_card"
    MAIN_INTERFACE = "main_interface"
    MULTI_SELECTION_COMBO_BOX = "multi_selection_combo_box"

    def path(self, theme=Theme.AUTO):
        theme = qconfig.theme if theme == Theme.AUTO else theme
        return f"src/ui/resource/qss/{theme.value.lower()}/{self.value}.qss"
