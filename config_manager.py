class _Config:

    def __init__(self, depth=0, **response):
        self._depth = depth
        for k, v in response.items():
            if isinstance(v,dict):
                _class = globals()[
                    "{}Config".format(
                        "".join(list(str(" ".join(list(k.split("_"))).title()).split(" ")))
                    )
                ]
                self.__dict__[k] = _class(self._depth+1, **v)
            else:
                self.__dict__[k] = v

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

    def __str__(self):
        _tab = "".join(["\t" for i in range(self._depth)]) if self._depth>0 else ""
        _str = str("")
        for k, v in self:
            if k == "_depth": 
                continue
            if isinstance(v, Config):
                _str += ("{}{} \n {}\n".format(_tab, k, str(v)))
            else:
                _str += ("{}{} : {}\n".format(_tab, k, v))
        return _str


class Config(_Config):
    def __init__(self, depth=0, **data):
        self.input = None
        self.save= None
        self.output = None
        self.deskew = None
        self.letters = None
        self.words = None
        self.dark_on_light = None
        self.swt_skip_edges = None
        self.gt = False
        super(Config, self).__init__(depth, **data)

class LettersConfig(_Config):
    def __init__(self, depth=0, **data):
        self.min_width = None
        self.min_height = None
        self.max_width = None
        self.max_height = None
        self.width_height_ratio = None
        self.height_width_ratio = None
        self.min_diagonal_mswt_ratio = None
        self.max_diagonal_mswt_ratio = None
        super(LettersConfig, self).__init__(depth, **data)

class WordsConfig(_Config):
    def __init__(self, depth=0, **data):
        self.thresh_pairs_y = None
        self.thresh_mswt = None
        self.thresh_height = None
        self.width_scale = None
        super(WordsConfig, self).__init__(depth, **data)

