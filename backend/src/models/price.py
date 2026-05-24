from pydantic import BaseModel
from pydantic.alias_generators import to_camel


class PriceOffer(BaseModel):
    source: str
    title: str
    price: str | None = None
    currency: str | None = None
    product_url: str | None = None
    image_url: str | None = None
    asin: str | None = None
    provider: str | None = None
    note: str | None = None

    model_config = {
        "alias_generator": to_camel,
        "populate_by_name": True,
    }
