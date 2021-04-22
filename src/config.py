from easydict import EasyDict as edict
from pprint import pformat
from fabric.cluster.configurator import Configurator
from fabric.deploy.sow import NodeTracer


_ERR_MSG_UNINITED_ = 'global config state not inited'


class _CFG():
    def __init__(self):
        self.manager = None
        self.settings = {}
        self.inited = False

    def init_state(self, cfg_yaml_path, override_opts):
        self.manager = Configurator(cfg_yaml_path)
        self.settings = edict(self.manager.config)
        if override_opts is not None:
            self.merge_from_list(override_opts)
        self.inited = True

    def __getattr__(self, name):
        assert self.inited, _ERR_MSG_UNINITED_
        return getattr(self.settings, name)

    def __str__(self):
        assert self.inited, _ERR_MSG_UNINITED_
        return pformat(self.settings)

    def merge_from_list(self, cfg_list):
        """Merge config (keys, values) in a list (e.g., from command line) into
        this cfg. For example, `cfg_list = ['FOO.BAR', 0.5]`.
        """
        assert len(cfg_list) % 2 == 0, \
            "Override list has odd length: {}; it must be a list of pairs".format(
                cfg_list
            )
        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            self.trace_and_replace(full_key, v)

    def trace_and_replace(self, key, val):
        tracer = NodeTracer(self.settings)
        tracer.advance_pointer(key)
        tracer.replace([], val)
        self.settings = tracer.state


_C = _CFG()
cfg = _C
