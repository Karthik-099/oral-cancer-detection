import { api, ApiError, onUnauthorized, tokenStore } from "@/lib/api";

const respond = (status: number, body: unknown) =>
  vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }));

afterEach(() => {
  tokenStore.set(null);
  vi.restoreAllMocks();
});

describe("api client", () => {
  it("sends the bearer token", async () => {
    tokenStore.set("abc");
    const fetchMock = respond(200, { status: "ok" });
    vi.stubGlobal("fetch", fetchMock);
    await api.health();
    const headers = fetchMock.mock.calls[0][1].headers as Headers;
    expect(headers.get("Authorization")).toBe("Bearer abc");
  });

  it("surfaces FastAPI string errors", async () => {
    vi.stubGlobal("fetch", respond(422, { detail: "This image does not appear to be an oral cavity photograph." }));
    await expect(api.models()).rejects.toMatchObject({ status: 422, message: expect.stringContaining("oral cavity") });
  });

  it("joins pydantic validation errors", async () => {
    vi.stubGlobal("fetch", respond(422, { detail: [{ msg: "Value error, bad email" }, { msg: "too short" }] }));
    await expect(api.login({ email: "x", password: "y" })).rejects.toThrow("bad email. too short");
  });

  it("notifies listeners on 401 when signed in", async () => {
    tokenStore.set("expired");
    const listener = vi.fn();
    const off = onUnauthorized(listener);
    vi.stubGlobal("fetch", respond(401, { detail: "Session expired" }));
    await expect(api.me()).rejects.toBeInstanceOf(ApiError);
    expect(listener).toHaveBeenCalledOnce();
    off();
  });

  it("reports network failures clearly", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fetch failed")));
    await expect(api.health()).rejects.toMatchObject({ status: 0 });
  });
});
