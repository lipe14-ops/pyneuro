class Polynomial(object):
    def __init__(self, degree: int, coeficients: list[float] = []) -> None:
        self.degree = degree
        self.terms_cache = {}
        self.coeficients = coeficients if coeficients else [ 1.0 ] * (self.degree + 1)

        self.set_all_coeficients(self.coeficients)

    def set_all_coeficients(self, coeficients: list[float]) -> None:
        if len(coeficients) != self.degree + 1:
            raise ValueError(f"this polynomial must have {self.degree + 1} coeficients.")

        self.coeficients = coeficients
        self.terms_cache.clear()

    def set_coeficient(self, position: int, value: float) -> None:
        self.coeficients[position] = value
        self.terms_cache.clear()

    def get_image_of(self, value: float | int) -> int | float:
        if self.terms_cache.get(value):
            return self.terms_cache[value]

        self.terms_cache[value] = sum(
                coeficient * pow(value, n)
                for n, coeficient in enumerate(self.coeficients)
                )

        return self.terms_cache[value]

