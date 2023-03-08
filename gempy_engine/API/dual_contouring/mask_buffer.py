class MaskBuffer:
    previous_mask = None

    @classmethod
    def clean(cls):
        cls.previous_mask = None
