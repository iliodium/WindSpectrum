from dataclasses import dataclass

from kivy.properties import StringProperty, BooleanProperty
from kivymd.uix.screen import MDScreen


@dataclass
class RoofWithEaves(MDScreen):
    _roof_type = StringProperty('flat')
    _roof_type_hidden = BooleanProperty(False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test(self, root, app):
        print(dir(root))
        print(dir(app))

    def hide_widget(self, *widgets):
        dohide = not self._roof_type_hidden
        for wid in widgets:
            if hasattr(wid, 'saved_attrs'):
                if not dohide:
                    wid.height, wid.size_hint_y, wid.opacity, wid.disabled = wid.saved_attrs
                    del wid.saved_attrs
            elif dohide:
                wid.saved_attrs = wid.height, wid.size_hint_y, wid.opacity, wid.disabled
                wid.height, wid.size_hint_y, wid.opacity, wid.disabled = 0, None, 0, True

        self._roof_type_hidden = dohide
