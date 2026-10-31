from importlib import import_module

try:
    import_module(".sampler.sglang.register", __name__)
    import_module(".patches", __name__)
except:
    pass
