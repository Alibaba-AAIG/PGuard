from src.model.modeling_pguard_ema import PluginGuard as PluginGuardEma
from src.model.modeling_pguard_gru import PluginGuard as PluginGuardGru
from src.model.modeling_pguard_lasttok import PluginGuard as PluginGuardLasttok

MODEL_MAP = {
    "ema": PluginGuardEma,
    "gru": PluginGuardGru,
    "lasttok": PluginGuardLasttok,
}
