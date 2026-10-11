import { pct, timeAgo, TIER_META } from "@/lib/utils";

describe("utils", () => {
  it("formats percentages and handles missing values", () => {
    expect(pct(0.705)).toBe("71%");
    expect(pct(0.705, 1)).toBe("70.5%");
    expect(pct(null)).toBe("–");
  });

  it("treats API timestamps as UTC", () => {
    const iso = new Date(Date.now() - 5 * 60_000).toISOString().replace("Z", "");
    expect(timeAgo(iso)).toBe("5m ago");
  });

  it("has metadata for every risk tier", () => {
    expect(Object.keys(TIER_META).sort()).toEqual(["high", "low", "moderate"]);
  });
});
