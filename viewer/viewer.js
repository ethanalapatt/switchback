"use strict";
/* Offline trace viewer.
 *
 * The replay check here is a deliberate reimplementation of
 * src/switchback/traces.py::replay_trace. A viewer that trusted the file it was
 * handed could draw a confident picture of an inconsistent trace, which is the
 * opposite of what a trace is for. Two independent implementations of the same
 * rules also mean a change to one of them has to be made twice on purpose.
 */

const TRACE_SCHEMA_VERSION = 1;

const state = { header: null, blocks: [], selected: null };

function element(id) {
  return document.getElementById(id);
}

function showError(message) {
  const box = element("error");
  box.textContent = message;
  box.classList.remove("hidden");
  element("report").classList.add("hidden");
}

function clearError() {
  element("error").classList.add("hidden");
}

function parseTrace(text) {
  const records = text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`line ${index + 1} is not valid JSON: ${error.message}`);
      }
    });
  if (records.length === 0) throw new Error("the file is empty");
  const [header, ...blocks] = records;
  if (header.record !== "header") throw new Error("the first line is not a trace header");
  if (header.schema_version !== TRACE_SCHEMA_VERSION) {
    throw new Error(
      `trace schema_version ${header.schema_version} is not ${TRACE_SCHEMA_VERSION}`
    );
  }
  blocks.forEach((block, index) => {
    if (block.record !== "block") throw new Error(`record ${index + 2} is not a block`);
  });
  return { header, blocks };
}

/* Mirrors replay_trace: block ids consecutive, rejection position equal to the
 * accepted count, accepted never above proposed, a speculative block committing
 * accepted+1 tokens, and both cache lengths at prompt+committed-1. A terminal
 * block may commit fewer tokens and leave its draft cache one short, because
 * ending on a committed EOS truncates the block and skips the catch-up call. */
function replay(header, blocks) {
  const problems = [];
  const prompt = header.prompt_tokens;
  let committed = 0;
  let accepted = 0;
  let proposed = 0;
  let previousDraft = null;

  blocks.forEach((block, index) => {
    const label = `block ${index}`;
    if (block.block_id !== index) {
      problems.push(`${label}: block_id is ${block.block_id}, expected ${index}`);
    }
    if (block.rejection_position !== null && block.rejection_position !== block.accepted) {
      problems.push(
        `${label}: rejection at ${block.rejection_position} but ${block.accepted} accepted`
      );
    }
    if (block.accepted > block.proposed) {
      problems.push(`${label}: accepted ${block.accepted} exceeds proposed ${block.proposed}`);
    }
    const terminal = Boolean(block.terminal);
    const startExpected = prompt + committed - 1;
    if (block.action === "speculative") {
      const expectedCommits = block.accepted + 1;
      if (terminal) {
        if (!(block.committed >= 1 && block.committed <= expectedCommits)) {
          problems.push(
            `${label}: terminal block committed ${block.committed}, expected 1..${expectedCommits}`
          );
        }
      } else if (block.committed !== expectedCommits) {
        problems.push(
          `${label}: committed ${block.committed} for ${block.accepted} accepted`
        );
      }
    }
    committed += block.committed;
    accepted += block.accepted;
    proposed += block.proposed;

    const expectedLength = prompt + committed - 1;
    if (block.target_cache_after !== expectedLength) {
      problems.push(
        `${label}: target cache is ${block.target_cache_after}, expected ${expectedLength}`
      );
    }
    const draft = block.draft_cache_after;
    if (draft !== null && draft !== undefined) {
      if (block.action === "speculative") {
        const lower = terminal ? expectedLength - 1 : expectedLength;
        if (draft < lower || draft > expectedLength) {
          problems.push(`${label}: draft cache is ${draft}, expected ${expectedLength}`);
        }
        if (previousDraft !== null && previousDraft !== startExpected) {
          problems.push(
            `${label}: drafted from a cache at ${previousDraft} when the block ` +
              `started at boundary ${startExpected}; a target-only step does not ` +
              `advance the draft cache`
          );
        }
      } else if (previousDraft !== null && draft !== previousDraft) {
        problems.push(
          `${label}: a target-only step moved the draft cache from ${previousDraft} to ${draft}`
        );
      }
    }
    previousDraft = draft === undefined ? null : draft;
  });

  if (committed !== header.output_ids.length) {
    problems.push(
      `blocks committed ${committed} tokens but the header records ${header.output_ids.length}`
    );
  }
  if (accepted !== header.accepted || proposed !== header.proposed) {
    problems.push(
      `block counters sum to ${accepted}/${proposed}, header says ` +
        `${header.accepted}/${header.proposed}`
    );
  }
  return { ok: problems.length === 0, problems, committed, accepted, proposed };
}

function stat(label, value) {
  return `<div class="stat"><div class="label">${label}</div><div class="value">${value}</div></div>`;
}

function blockClass(block) {
  if (block.bypass_reason) return "bypass";
  if (block.action !== "speculative") return "plain";
  return block.rejection_position === null ? "accepted" : "rejected";
}

function render() {
  const { header, blocks } = state;
  const verdict = replay(header, blocks);
  const speculative = blocks.filter((block) => block.action === "speculative");
  const rejected = speculative.filter((block) => block.rejection_position !== null);
  const totalMs = blocks.reduce((sum, block) => sum + block.duration_ns, 0) / 1e6;

  element("summary").innerHTML = [
    stat("Engine", header.engine),
    stat("Mode", `${header.mode}${header.mode === "sample" ? ` @ ${header.temperature}` : ""}`),
    stat("Draft length", header.gamma === null ? "none" : header.gamma),
    stat("Prompt tokens", header.prompt_tokens),
    stat("Output tokens", header.output_ids.length),
    stat("Terminated on", header.termination),
    stat("Target calls", header.target_calls),
    stat("Draft calls", header.draft_calls),
    stat(
      "Acceptance",
      header.proposed ? `${header.accepted}/${header.proposed}` : "n/a"
    ),
    stat("Blocks", `${speculative.length} speculative, ${rejected.length} rejected`),
    stat("Sum of block time", `${totalMs.toFixed(1)} ms`),
  ].join("");

  const verdictNode = element("verdict");
  verdictNode.innerHTML = verdict.ok
    ? '<span class="verdict ok">Replay passed.</span> Cache lengths are consistent with the tokens committed at every block boundary.'
    : `<span class="verdict bad">Replay failed.</span> ${verdict.problems
        .map((problem) => `<br>&nbsp;&nbsp;${problem}`)
        .join("")}`;

  element("provenance").textContent =
    `Source commit ${header.source.commit || "unknown"}` +
    (header.source.dirty ? " (tree was dirty)" : "") +
    `. Prompt sha256 ${header.prompt_sha256.slice(0, 16)}…, ` +
    `${header.prompt_tokens} tokens. The prompt text is not stored in the trace.`;

  element("blocks").innerHTML = blocks
    .map((block, index) => {
      const label =
        block.action === "speculative"
          ? `${index}: g${block.gamma} ${block.accepted}/${block.proposed}`
          : `${index}: ${block.action}`;
      return `<button type="button" class="block ${blockClass(block)}" data-index="${index}">${label}</button>`;
    })
    .join("");

  const counts = new Map();
  speculative.forEach((block) => {
    const key = block.rejection_position === null ? "none" : String(block.rejection_position);
    counts.set(key, (counts.get(key) || 0) + 1);
  });
  const ordered = [...counts.entries()].sort((a, b) => {
    if (a[0] === "none") return 1;
    if (b[0] === "none") return -1;
    return Number(a[0]) - Number(b[0]);
  });
  element("positions").querySelector("tbody").innerHTML = ordered
    .map(
      ([key, count]) =>
        `<tr><td class="left">${key === "none" ? "none (all accepted)" : key}</td>` +
        `<td>${count}</td><td>${((100 * count) / speculative.length).toFixed(1)}%</td></tr>`
    )
    .join("");

  element("table").querySelector("tbody").innerHTML = blocks
    .map(
      (block, index) =>
        `<tr data-index="${index}">` +
        `<td class="left">${index}</td><td class="left">${block.action}</td>` +
        `<td>${block.gamma || ""}</td><td>${block.proposed}</td><td>${block.accepted}</td>` +
        `<td>${block.rejection_position === null ? "—" : block.rejection_position}</td>` +
        `<td>${block.committed}</td><td>${block.target_cache_after}</td>` +
        `<td>${block.draft_cache_after === null ? "—" : block.draft_cache_after}</td>` +
        `<td>${(block.duration_ns / 1e6).toFixed(2)}</td></tr>`
    )
    .join("");

  element("report").classList.remove("hidden");
  select(blocks.findIndex((block) => block.rejection_position !== null) >= 0
    ? blocks.findIndex((block) => block.rejection_position !== null)
    : 0);
}

function select(index) {
  const { header, blocks } = state;
  if (index < 0 || index >= blocks.length) return;
  state.selected = index;
  const block = blocks[index];

  let committedBefore = 0;
  for (let i = 0; i < index; i += 1) committedBefore += blocks[i].committed;
  const committedTokens = header.output_ids.slice(
    committedBefore,
    committedBefore + block.committed
  );

  const parts = [];
  if (block.action === "speculative") {
    const accepted = block.accepted;
    const rejectedCount = block.proposed - accepted;
    for (let i = 0; i < accepted; i += 1) {
      parts.push(`<span class="token accepted" title="accepted draft candidate ${i}">${committedTokens[i]}</span>`);
    }
    for (let i = 0; i < rejectedCount; i += 1) {
      parts.push(
        `<span class="token rejected" title="proposed but discarded">candidate ${accepted + i}</span>`
      );
    }
    if (committedTokens.length > accepted) {
      const last = committedTokens[committedTokens.length - 1];
      const kind = block.rejection_position === null ? "bonus from the target" : "target correction";
      parts.push(`<span class="token correction" title="${kind}">${last}</span>`);
    }
  } else {
    committedTokens.forEach((token) => {
      parts.push(`<span class="token correction" title="target-only step">${token}</span>`);
    });
  }

  const sentence =
    block.action !== "speculative"
      ? block.bypass_reason
        ? `Bypass: ${block.bypass_reason}`
        : `A ${block.action} step committed ${block.committed} token(s) from the target alone.`
      : block.rejection_position === null
        ? `All ${block.proposed} candidates were accepted, and the target added a bonus token, ` +
          `so ${block.committed} tokens were committed from one target forward call. ` +
          `The draft then needed a catch-up call to reach the same boundary.`
        : `Candidate ${block.rejection_position} disagreed with the target. That candidate and ` +
          `every proposal after it were discarded, the target's own token was committed in its ` +
          `place, and both caches were cropped to ${block.target_cache_after}.`;

  element("detail").innerHTML =
    `<div class="grid">` +
    stat("Block", index) +
    stat("Action", block.action) +
    stat("Draft length", block.gamma || "—") +
    stat("Accepted", `${block.accepted}/${block.proposed}`) +
    stat("First rejection", block.rejection_position === null ? "none" : block.rejection_position) +
    stat("Committed", block.committed) +
    stat("Target cache after", block.target_cache_after) +
    stat("Draft cache after", block.draft_cache_after === null ? "—" : block.draft_cache_after) +
    stat("Block time", `${(block.duration_ns / 1e6).toFixed(2)} ms`) +
    `</div>` +
    `<p class="note">${sentence}</p>` +
    `<div class="tokens">${parts.join("")}</div>` +
    `<div class="legend">` +
    `<span><i class="swatch" style="background:var(--accept-soft);color:var(--accept)"></i> accepted draft token</span>` +
    `<span><i class="swatch" style="background:var(--reject-soft);color:var(--reject)"></i> proposed then discarded</span>` +
    `<span><i class="swatch" style="background:var(--correct-soft);color:var(--correct)"></i> token the target itself produced</span>` +
    `</div>`;

  document.querySelectorAll(".block").forEach((node) => {
    node.classList.toggle("selected", Number(node.dataset.index) === index);
  });
  document.querySelectorAll("#table tbody tr").forEach((node) => {
    node.classList.toggle("selected", Number(node.dataset.index) === index);
  });
}

function load(text) {
  try {
    clearError();
    const parsed = parseTrace(text);
    state.header = parsed.header;
    state.blocks = parsed.blocks;
    render();
  } catch (error) {
    showError(`Could not read that trace: ${error.message}`);
  }
}

document.addEventListener("click", (event) => {
  const target = event.target.closest("[data-index]");
  if (target) select(Number(target.dataset.index));
});

element("file").addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => load(String(reader.result));
  reader.onerror = () => showError("Could not read that file.");
  reader.readAsText(file);
});

element("demo").addEventListener("click", () => {
  if (typeof window.SWITCHBACK_EXAMPLE_TRACE === "string") {
    load(window.SWITCHBACK_EXAMPLE_TRACE);
  } else {
    showError(
      "No built-in example is bundled. Pick a trace written by " +
        "`python -m switchback trace --out artifacts/traces/greedy_g4.jsonl`."
    );
  }
});
