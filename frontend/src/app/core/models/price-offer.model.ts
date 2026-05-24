export interface PriceOffer {
  source: string;
  title: string;
  price?: string | null;
  currency?: string | null;
  productUrl?: string | null;
  imageUrl?: string | null;
  asin?: string | null;
  provider?: string | null;
  note?: string | null;
}
