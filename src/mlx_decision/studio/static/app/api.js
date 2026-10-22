// The Studio's own API: JSON in and out, errors as ApiError with the API's fields.

export class ApiError extends Error {
  constructor(status, body) {
    const error = (body && body.error) || {};
    super(
      error.message ||
        (status === 0
          ? "the Studio is not answering"
          : status >= 500
            ? `the Studio failed (HTTP ${status}); the terminal running mlx-decision web says why`
            : `the Studio answered ${status}`),
    );
    this.status = status;
    this.param = error.param ?? null;
    this.code = error.code ?? null;
    this.current = error.current;
    this.body = body; // the whole error body (an upload's errors by line)
  }
}

export async function api(path, { method = "GET", body, signal } = {}) {
  const init = { method, headers: { Accept: "application/json" }, signal };
  if (method !== "GET" && method !== "HEAD") {
    // Every write is JSON, a DELETE included: the Studio refuses anything else.
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body ?? {});
  }
  let response;
  try {
    response = await fetch(`/studio/api${path}`, init);
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError(0, null); // no answer: the Studio is not running
  }
  if (response.status === 204) return null;
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new ApiError(response.status, data);
  return data;
}
