# -*- coding: utf-8 -*-
"""
Minimal logic backing the MyBlankGui. Centralizes POI parsing/building and
exposes a simple .ping() method for demo purposes.

Example config:

logic:
  my_blank_logic:
    module.Class: 'my_blank_logic.MyBlankLogic'
"""

from PySide2 import QtCore
from qudi.core.module import LogicBase


class MyBlankLogic(LogicBase):
    # Signals to update GUI asynchronously
    sigStatus = QtCore.Signal(str)
    sigLog = QtCore.Signal(str)

    def on_activate(self):
        self.sigStatus.emit('Idle')

    def on_deactivate(self):
        pass

    # --- Demo API -----------------------------------------------------------
    @QtCore.Slot(result=str)
    def ping(self) -> str:
        """Simple demo call from GUI."""
        self.sigLog.emit('Logic received ping()')
        self.sigStatus.emit('Pinged')
        return 'pong'

    # --- POI helpers --------------------------------------------------------

    @QtCore.Slot(str, int, int, result='QVariant')
    def build_pois_simple(self, prefix: str, start: int, end: int, width: int = 0):
        """Return list of {'poi': '<prefix><number>'}.

        - No zero-padding if width == 0
        - If width > 0, numbers are zero-padded to the given width
        """
        prefix = (prefix or "").strip()
        if not prefix or end < start:
            return []
        fmt = f"{{:0{int(width)}d}}" if int(width) > 0 else "{:d}"
        return [{'poi': f"{prefix}{fmt.format(i)}"} for i in range(int(start), int(end) + 1)]

    @QtCore.Slot(str, result='QVariant')
    def parse_pois_advanced(self, text: str):
        """Parse tokens like 'POI1-100, TEST1-3, NV001-010, NV1' into
        [{'poi': ...}, ...].

        - Supports single IDs and ranges with shared prefix
        - For ranges, infers zero-padding width from either bound
        - Accepts hyphen, en dash, or em dash
        - De-duplicates while preserving order
        """
        import re

        # prefix + number, optional whitespace, dash variant, optional whitespace, number
        rng_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_\-]*?)(\d+)\s*[-\u2013\u2014]\s*(\d+)$")
        out, seen = [], set()
        tokens = [t.strip() for t in (text or "").split(",") if t.strip()]
        for token in tokens:
            m = rng_re.match(token)
            if m:
                prefix, a, b = m.group(1), m.group(2), m.group(3)
                start, end = int(a), int(b)
                if end < start:
                    continue
                width = max(len(a), len(b)) if (a.startswith('0') or b.startswith('0')) else 0
                fmt = f"{{:0{width}d}}" if width > 0 else "{:d}"
                for i in range(start, end + 1):
                    poi = f"{prefix}{fmt.format(i)}"
                    if poi not in seen:
                        seen.add(poi)
                        out.append({'poi': poi})
            else:
                if token not in seen:
                    seen.add(token)
                    out.append({'poi': token})
        return out

    # Convenience for preview text
    def preview_text(self, poi_dicts):
        n = len(poi_dicts)
        if n == 0:
            return "0 POIs"
        if n <= 6:
            sample = ", ".join(d['poi'] for d in poi_dicts)
        else:
            sample = f"{poi_dicts[0]['poi']}, {poi_dicts[1]['poi']}, …, {poi_dicts[-2]['poi']}, {poi_dicts[-1]['poi']}"
        return f"{n} POIs: {sample}"

# Backwards-compatibility alias for configs expecting class name "MyBlank"
MyBlank = MyBlankLogic
