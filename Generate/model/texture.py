class MoonTexture:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.texture_id = None
        self.texture_bytes = None

    def check_parameters(self):
        assert type(self.width) == int and type(self.height) == int
        assert self.width > 0 and self.height > 0
        assert self.texture_bytes is not None
