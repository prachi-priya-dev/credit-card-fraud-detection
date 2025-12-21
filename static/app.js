function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const text = await res.text();
  try {
    const json = JSON.parse(text);
    if (!res.ok) {
      throw new Error(json.detail || json.error || "Request failed");
    }
    return json;
  } catch {
    // If backend returned HTML or non-json, show it
    throw new Error(`Non-JSON response: ${text.slice(0, 180)}...`);
  }
}

document.getElementById("predictBtn").addEventListener("click", async () => {
  const apiUrl = document.getElementById("apiUrl").value.trim().replace(/\/$/, "");
  const threshold = Number(document.getElementById("threshold").value);
  const resultEl = document.getElementById("result");

  let payload;
  try {
    payload = JSON.parse(document.getElementById("payload").value);
  } catch (e) {
    resultEl.className = "result error";
    resultEl.textContent = "Invalid JSON in input box.";
    return;
  }

  resultEl.className = "result muted";
  resultEl.textContent = "Predicting…";

  try {
    // Your API should support threshold either in query or body.
    // We'll send it in body for simplicity.
    const data = await postJson(`${apiUrl}/predict`, {
      ...payload,
      threshold,
    });

    resultEl.className = "result ok";
    resultEl.textContent = pretty(data);
  } catch (e) {
    resultEl.className = "result error";
    resultEl.textContent = e.message;
  }
});
