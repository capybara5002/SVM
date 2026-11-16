"use strict";

const $ = (selector) => document.querySelector(selector);
const integer = new Intl.NumberFormat("vi-VN");
const decimal = (value, digits = 5) => Number(value).toFixed(digits);
const signed = (value) => `${value >= 0 ? "+" : ""}${decimal(value)}`;
const stages = [
  ["input", "Đọc nội dung", "Tiếp nhận văn bản đầu vào"],
  ["clean", "Làm sạch văn bản", "Chuẩn hóa theo pipeline huấn luyện"],
  ["segment", "Tách từ tiếng Việt", "Chuẩn bị đầu vào cho TF-IDF"],
  ["tfidf", "Vector hóa TF-IDF", "Biểu diễn văn bản bằng trọng số"],
  ["svm", "Dự đoán SVM", "Tính điểm quyết định và nhãn"],
];
let mode = "text";
let selectedFile = null;
let catalog = null;
let busy = false;
let report = null;
let activeStage = null;
let completeCount = 0;
const stageElements = new Map();

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}
function announce(text) { $("#live-status").textContent = text; }
function showError(target, text) {
  $(target).textContent = text;
  $(target).hidden = !text;
}
function resetOutput() {
  if (busy) return;
  report = null;
  activeStage = null;
  completeCount = 0;
  stageElements.clear();
  $("#completed-count").textContent = "0/5";
  $("#result").hidden = true;
  $("#result").replaceChildren();
  $("#pipeline").hidden = true;
  $("#pipeline").replaceChildren();
  $("#empty-state").hidden = false;
  $("#download").disabled = true;
  showError("#analysis-error", "");
  showError("#input-error", "");
  document.querySelectorAll("[data-step]").forEach((node) => { node.className = ""; });
}
function setBusy(value) {
  busy = value;
  $("#analysis-form").querySelectorAll("button, input, textarea, select").forEach((node) => {
    node.disabled = value;
  });
  $("#analyze").disabled = value || !catalog?.models.length;
  $("#model").disabled = value || !catalog?.models.length;
  $("#analyze-label").textContent = value ? "Đang phân tích…" : "Phân tích nội dung";
  $("#analysis-form").setAttribute("aria-busy", String(value));
  $("#pipeline").setAttribute("aria-busy", String(value));
}
function setMode(next) {
  if (busy) return;
  mode = next;
  resetOutput();
  document.querySelectorAll("[data-mode]").forEach((button) => {
    const selected = button.dataset.mode === mode;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
  });
  $("#text-panel").hidden = mode !== "text";
  $("#file-panel").hidden = mode !== "file";
}
function countCharacters() {
  $("#char-count").textContent = `${integer.format($("#article").value.length)} ký tự`;
}
function selectFile(file) {
  if (busy) return;
  resetOutput();
  if (!file) return;
  selectedFile = null;
  $("#selected-file").hidden = true;
  if (!/\.(txt|md|docx|pdf)$/i.test(file.name)) {
    showError("#input-error", "Chọn tệp TXT, MD, DOCX hoặc PDF có văn bản.");
    return;
  }
  if (file.size > 5 * 1024 * 1024) {
    showError("#input-error", "Tệp vượt quá giới hạn 5 MB.");
    return;
  }
  selectedFile = file;
  $("#selected-file").hidden = false;
  $("#file-name").textContent = file.name;
  $("#file-size").textContent = `${integer.format(file.size)} byte · Sẵn sàng phân tích`;
}
function modelDescription() {
  const model = catalog?.models.find((item) => item.id === $("#model").value);
  if (!model) return;
  $("#model-description").textContent = `${model.segmented ? "Có" : "Không"} tách từ Underthesea${model.cv_f1 !== null ? ` · F1 nhãn 1 qua 5-fold CV: ${decimal(model.cv_f1, 4)}` : ""}.`;
}
function startPipeline() {
  $("#empty-state").hidden = true;
  $("#pipeline").hidden = false;
  stageElements.clear();
  stages.forEach(([id, title, subtitle], index) => {
    const details = el("details", "stage");
    details.id = `stage-${id}`;
    const summary = el("summary");
    const badge = el("span", "stage-badge");
    badge.append(el("span", "", String(index + 1).padStart(2, "0")));
    const label = el("div", "stage-title");
    label.append(el("strong", "", title), el("small", "", subtitle));
    const time = el("span", "stage-time", "Chờ xử lý");
    const chevron = el("span", "stage-chevron", "⌄");
    chevron.setAttribute("aria-hidden", "true");
    summary.append(badge, label, time, chevron);
    const body = el("div", "stage-body");
    details.append(summary, body);
    $("#pipeline").append(details);
    stageElements.set(id, { details, body, time, subtitle: label.lastChild });
  });
}
function chips(container, entries) {
  const row = el("div", "data-chips");
  entries.forEach((text) => row.append(el("span", "data-chip", text)));
  container.append(row);
}
function output(container, text) { container.append(el("div", "text-output", text)); }

function dataTable(container, rows, columns, { search = false, title = "", pageSize = 12 } = {}) {
  let page = 0;
  let filtered = rows;
  const toolbar = el("div", "table-toolbar");
  const caption = el("span", "", title);
  toolbar.append(caption);
  if (search) {
    const input = el("input", "feature-search");
    input.type = "search";
    input.placeholder = "Tìm từ đặc trưng…";
    input.setAttribute("aria-label", "Tìm kiếm đặc trưng TF-IDF");
    input.addEventListener("input", () => {
      const query = input.value.trim().toLocaleLowerCase("vi");
      filtered = rows.filter((row) => row.term.toLocaleLowerCase("vi").includes(query));
      page = 0;
      render();
    });
    toolbar.append(input);
  }
  const wrapper = el("div", "table-wrapper");
  const table = el("table");
  const head = el("thead");
  const headRow = el("tr");
  columns.forEach((column) => {
    const th = el("th", "", column.label);
    th.scope = "col";
    headRow.append(th);
  });
  head.append(headRow);
  const body = el("tbody");
  table.append(head, body);
  wrapper.append(table);
  const pagination = el("div", "pagination");
  const previous = el("button", "", "‹");
  previous.type = "button";
  previous.setAttribute("aria-label", "Trang trước");
  const info = el("span");
  const next = el("button", "", "›");
  next.type = "button";
  next.setAttribute("aria-label", "Trang sau");
  previous.addEventListener("click", () => { page--; render(); });
  next.addEventListener("click", () => { page++; render(); });
  pagination.append(previous, info, next);
  container.append(toolbar, wrapper, pagination);
  function render() {
    body.replaceChildren();
    const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
    const displayed = filtered.slice(page * pageSize, (page + 1) * pageSize);
    displayed.forEach((row) => {
      const tr = el("tr");
      columns.forEach((column) => {
        const value = row[column.key];
        const td = el("td", column.numeric ? "number" : "", column.format ? column.format(value) : String(value));
        if (column.signed) td.classList.add(value >= 0 ? "positive" : "negative");
        tr.append(td);
      });
      body.append(tr);
    });
    if (!displayed.length) {
      const tr = el("tr");
      const td = el("td", "", "Không có đặc trưng khớp với từ khóa.");
      td.colSpan = columns.length;
      tr.append(td);
      body.append(tr);
    }
    info.textContent = `${filtered.length} dòng · ${page + 1}/${totalPages}`;
    previous.disabled = page === 0;
    next.disabled = page + 1 >= totalPages;
  }
  render();
}

function renderStage(id, data, container) {
  if (id === "input") {
    chips(container, [`${integer.format(data.characters)} ký tự`, `${integer.format(data.whitespace_words)} đơn vị cách trắng`, data.source.filename || "Nhập trực tiếp"]);
    if (data.source.pages) container.append(el("p", "mini-title", `Đã trích xuất ${data.source.pages} trang PDF`));
    output(container, data.text);
  } else if (id === "clean") {
    chips(container, [`${integer.format(data.characters)} ký tự sau làm sạch`, "Giữ dấu tiếng Việt, phủ định và dấu câu"]);
    output(container, data.text);
    container.append(el("p", "mini-title", "6 thao tác làm sạch · Mở để xem văn bản sau từng thao tác"));
    data.operations.forEach((operation) => {
      const details = el("details", "operation");
      const summary = el("summary", "", operation.name);
      summary.append(el("span", "operation-status", operation.changed ? "Đã thay đổi" : "Không đổi"));
      details.append(summary);
      output(details, operation.text);
      container.append(details);
    });
  } else if (id === "segment") {
    container.append(el("p", "", data.note));
    output(container, data.text);
    container.append(el("p", "mini-title", `${integer.format(data.token_count)} đơn vị sau tách theo khoảng trắng`));
    const tokens = el("div", "tokens");
    data.tokens.forEach((token) => tokens.append(el("span", "token", token)));
    container.append(tokens);
  } else if (id === "tfidf") {
    chips(container, [`${data.active_features}/${integer.format(data.vocabulary_size)} đặc trưng khác 0`, `${data.oov.length} từ khác nhau ngoài từ vựng`, `Chuẩn L2: ${decimal(data.l2_norm, 3)}`]);
    container.append(el("p", "formula", data.formula));
    dataTable(container, data.features, [
      { key: "term", label: "Đặc trưng" },
      { key: "tf", label: "TF (lần)", numeric: true },
      { key: "idf", label: "IDF", numeric: true, format: (v) => decimal(v, 3) },
      { key: "weight", label: "TF-IDF ↓", numeric: true, format: (v) => decimal(v, 5) },
    ], { search: true, title: "Toàn bộ đặc trưng, xếp theo TF-IDF" });
    if (data.oov.length) {
      const details = el("details", "operation");
      details.append(el("summary", "", `${data.oov.length} từ không có trong từ vựng đã học (bị bỏ qua)`));
      details.append(el("p", "oov-text", data.oov.map((item) => `${item.term} (${item.count})`).join(" · ")));
      container.append(details);
    }
  } else if (id === "svm") {
    const explanation = data.explanation;
    chips(container, [`Kernel ${data.parameters.kernel.toUpperCase()}`, `C = ${data.parameters.C}`, `class_weight = ${data.parameters.class_weight || "None"}`, `${data.support_vectors} vector hỗ trợ`]);
    container.append(el("p", "formula", explanation.formula));
    const summary = el("div", "svm-summary");
    [["Tổng đóng góp", signed(explanation.contribution_sum)], ["Hệ số chặn b", signed(data.bias)], ["Điểm f(x)", signed(data.decision_score)], ["Nhãn dự đoán", `${data.label} · ${data.label === 1 ? "unreliable" : "reliable"}`]].forEach(([label, value]) => {
      const cell = el("div", "", label);
      cell.append(el("b", "", value));
      summary.append(cell);
    });
    container.append(summary, el("p", "", "Điểm âm nghiêng về nhãn 0; điểm dương nghiêng về nhãn 1. Nhãn cuối do SVC.predict trả về. Điểm này không phải xác suất hay phần trăm tin cậy."));
    if (explanation.kind === "linear") {
      dataTable(container, explanation.terms, [
        { key: "term", label: "Đặc trưng" },
        { key: "coefficient", label: "Hệ số w", numeric: true, format: (v) => decimal(v, 4) },
        { key: "contribution", label: "w × TF-IDF", numeric: true, signed: true, format: signed },
      ], { title: "Đóng góp từng từ · Xếp theo trị tuyệt đối" });
    } else {
      container.append(el("p", "mini-title", `γ = ${explanation.gamma} · Phân rã theo vector hỗ trợ`), el("p", "", explanation.note));
      dataTable(container, explanation.top_support_vectors, [
        { key: "index", label: "Chỉ số SV" },
        { key: "similarity", label: "K(x, SV)", numeric: true, format: (v) => decimal(v, 5) },
        { key: "contribution", label: "Đóng góp", numeric: true, signed: true, format: signed },
      ], { title: "Vector hỗ trợ có |đóng góp| lớn nhất", pageSize: 10 });
    }
    container.append(el("p", "mini-title", `Tệp mô hình: ${data.model.artifact}`));
  }
}

function renderResult(data, duration) {
  const card = $("#result");
  card.replaceChildren();
  card.className = `result-card${data.label === 1 ? " unreliable" : ""}`;
  const top = el("div", "result-top");
  top.append(el("span", "", "KẾT QUẢ DỰ ĐOÁN"), el("span", "", `${decimal(duration / 1000, 2)} giây · Mô hình ${data.model.id}`));
  const verdict = el("h3", "result-verdict", data.label_name);
  const row = el("div", "result-score-row");
  const score = el("div", "", "Điểm quyết định SVM");
  score.append(el("span", "score-number", signed(data.decision_score)));
  row.append(score, el("p", "", `Nhãn ${data.label} (${data.label === 1 ? "unreliable" : "reliable"}). Ngưỡng quyết định: 0. Đây là điểm của mô hình, không phải phần trăm tin cậy.`));
  card.append(top, verdict, row);
  data.warnings.forEach((warning) => card.append(el("p", "warning", `ⓘ ${warning}`)));
  card.append(el("p", "result-note", data.disclaimer));
  card.hidden = false;
}
function handleEvent(event) {
  if (event.type === "error") throw new Error(event.error);
  if (event.type === "stage") {
    const view = stageElements.get(event.stage);
    if (!view) return;
    const nav = document.querySelector(`[data-step="${event.stage}"]`);
    if (event.status === "running") {
      activeStage = event.stage;
      view.details.className = "stage running";
      view.time.textContent = "Đang xử lý…";
      nav.className = "running";
      announce(`Đang ${stages.find((stage) => stage[0] === event.stage)[1].toLocaleLowerCase("vi")}`);
    } else {
      view.details.className = "stage complete";
      view.time.textContent = `${decimal(event.duration_ms, 1)} ms`;
      view.subtitle.textContent = "Hoàn tất · Mở để xem chi tiết";
      view.body.replaceChildren();
      renderStage(event.stage, event.data, view.body);
      view.details.open = event.stage !== "input";
      nav.className = "complete";
      activeStage = null;
      completeCount++;
      $("#completed-count").textContent = `${completeCount}/5`;
    }
  } else if (event.type === "complete") {
    report = event;
    renderResult(event.result, event.duration_ms);
    $("#download").disabled = false;
    announce(`Đã phân tích xong. Dự đoán: ${event.result.label_name}.`);
  }
}

async function analyze(event) {
  event.preventDefault();
  if (busy || !catalog?.models.length) return;
  resetOutput();
  if (mode === "text" && !$("#article").value.trim()) {
    showError("#input-error", "Hãy dán nội dung bài viết trước khi phân tích.");
    $("#article").focus();
    return;
  }
  if (mode === "file" && !selectedFile) {
    showError("#input-error", "Hãy chọn một tệp để phân tích.");
    return;
  }
  const form = new FormData();
  form.append("mode", mode);
  form.append("model", $("#model").value);
  if (mode === "text") form.append("text", $("#article").value);
  else form.append("file", selectedFile, selectedFile.name);
  setBusy(true);
  startPipeline();
  try {
    const response = await fetch("/api/analyze", { method: "POST", body: form });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || "Không thể gửi nội dung để phân tích.");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop();
        lines.filter((line) => line.trim()).forEach((line) => handleEvent(JSON.parse(line)));
        if (done) break;
      }
      if (buffer.trim()) handleEvent(JSON.parse(buffer));
      if (!report) throw new Error("Kết nối bị gián đoạn trước khi phân tích xong. Hãy thử lại.");
    } finally { reader.releaseLock(); }
  } catch (error) {
    showError("#analysis-error", error instanceof TypeError ? "Không kết nối được server. Kiểm tra cửa sổ chạy run_web.py rồi thử lại." : error.message);
    if (activeStage) {
      const view = stageElements.get(activeStage);
      view.details.className = "stage failed";
      view.time.textContent = "Chưa hoàn tất";
      document.querySelector(`[data-step="${activeStage}"]`).className = "failed";
    }
    announce("Phân tích chưa hoàn tất. Xem thông báo lỗi.");
  } finally { setBusy(false); }
}

$("#analysis-form").addEventListener("submit", analyze);
$("#article").addEventListener("input", () => { countCharacters(); resetOutput(); });
$("#clear").addEventListener("click", () => { $("#article").value = ""; countCharacters(); resetOutput(); $("#article").focus(); });
$("#sample").addEventListener("click", () => {
  $("#article").value = "THÔNG BÁO VỀ HOẠT ĐỘNG THƯ VIỆN\n\nThư viện trường thông báo sẽ mở cửa từ 7 giờ 30 đến 17 giờ vào các ngày trong tuần. Sinh viên có thể tra cứu tài liệu, mượn sách và sử dụng phòng tự học theo nội quy. Lịch hoạt động trong kỳ thi sẽ được cập nhật trên trang thông tin chính thức của trường.\n\nĐây là văn bản mẫu tự soạn để thử quy trình, không có nhãn kiểm chứng.";
  countCharacters(); resetOutput(); $("#article").focus();
});
document.querySelectorAll("[data-mode]").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
  button.addEventListener("keydown", (event) => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      setMode(event.key === "Home" ? "text" : event.key === "End" ? "file" : mode === "text" ? "file" : "text");
      document.querySelector(`[data-mode="${mode}"]`).focus();
    }
  });
});
$("#upload").addEventListener("change", (event) => selectFile(event.target.files[0]));
$("#remove-file").addEventListener("click", () => { selectedFile = null; $("#upload").value = ""; $("#selected-file").hidden = true; resetOutput(); });
["dragenter", "dragover"].forEach((name) => $("#dropzone").addEventListener(name, (event) => { event.preventDefault(); if (!busy) $("#dropzone").classList.add("dragging"); }));
["dragleave", "drop"].forEach((name) => $("#dropzone").addEventListener(name, (event) => { event.preventDefault(); $("#dropzone").classList.remove("dragging"); }));
$("#dropzone").addEventListener("drop", (event) => {
  if (busy) return;
  if (event.dataTransfer.files.length !== 1) { showError("#input-error", "Mỗi lần chọn một tệp để phân tích."); return; }
  selectFile(event.dataTransfer.files[0]);
});
$("#model").addEventListener("change", () => { modelDescription(); resetOutput(); });
document.addEventListener("keydown", (event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); $("#analysis-form").requestSubmit(); } });
$("#download").addEventListener("click", () => {
  if (!report) return;
  const data = { ...report, exported_at: new Date().toISOString() };
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json;charset=utf-8" }));
  const link = el("a");
  link.href = url;
  link.download = `svm-analysis-${Date.now()}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

async function initialize() {
  try {
    const response = await fetch("/api/models");
    if (!response.ok) throw new Error("Không tải được danh sách mô hình.");
    catalog = await response.json();
    if (!catalog.models.length) throw new Error("Chưa có mô hình. Xem hướng dẫn huấn luyện trong README.");
    $("#model").replaceChildren();
    catalog.models.forEach((model) => {
      const option = el("option", "", `${model.name}${model.id === catalog.default ? " · Đề xuất theo CV" : ""}`);
      option.value = model.id;
      $("#model").append(option);
    });
    $("#model").value = catalog.default;
    modelDescription();
    $("#connection").replaceChildren(el("span"), document.createTextNode(`${catalog.models.length} mô hình sẵn sàng`));
    setBusy(false);
  } catch (error) {
    $("#connection").classList.add("offline");
    $("#connection").replaceChildren(el("span"), document.createTextNode("Chưa kết nối mô hình"));
    showError("#input-error", error.message);
  }
}
initialize();
