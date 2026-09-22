// Lucide icons (ISC), inlined so the example needs no icon dependency.
const CHEVRON = "<svg class=\"icon\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path d=\"m6 9 6 6 6-6\" /></svg>";
const CHECK = "<svg class=\"icon\" viewBox=\"0 0 24 24\" aria-hidden=\"true\"><path d=\"M20 6 9 17l-5-5\" /></svg>";

/**
 * Replaces a native select's popup with a listbox styled like shadcn's Select. The hidden `<select>` stays the
 * source of truth: it keeps the value, the `change` events and the `disabled` state the rest of the page uses.
 */
export function enhanceSelect(select: HTMLSelectElement) {
  const trigger = document.createElement("button");
  trigger.type = "button";
  trigger.id = `${select.id}-trigger`;
  trigger.className = "select select-trigger";
  trigger.setAttribute("aria-haspopup", "listbox");
  trigger.setAttribute("aria-expanded", "false");
  const label = document.createElement("span");
  trigger.append(label);
  trigger.insertAdjacentHTML("beforeend", CHEVRON);

  // The popover sits in the top layer, so the sidebar's scrolling cannot clip it.
  const list = document.createElement("div");
  list.id = `${select.id}-listbox`;
  list.className = "select-content";
  list.popover = "auto";
  list.tabIndex = -1;
  list.setAttribute("role", "listbox");
  trigger.popoverTargetElement = list;
  trigger.setAttribute("aria-controls", list.id);

  const items = Array.from(select.options, (option, index) => {
    const item = document.createElement("div");
    item.id = `${list.id}-${index}`;
    item.className = "select-item";
    item.setAttribute("role", "option");
    item.textContent = option.text;
    item.insertAdjacentHTML("beforeend", CHECK);
    item.addEventListener("pointermove", () => highlight(index));
    item.addEventListener("click", () => choose(index));

    return item;
  });

  list.append(...items);

  for (const element of select.labels) {
    element.htmlFor = trigger.id;
    list.setAttribute("aria-label", element.textContent ?? "");
  }

  select.hidden = true;
  select.after(trigger);
  document.body.append(list);

  let active = 0;

  function sync() {
    label.textContent = select.selectedOptions[0]?.text ?? "";
    trigger.disabled = select.disabled;
    items.forEach((item, index) => item.setAttribute("aria-selected", String(index === select.selectedIndex)));
  }

  function highlight(index: number) {
    active = Math.max(0, Math.min(items.length - 1, index));
    items.forEach((item, other) => item.classList.toggle("is-active", other === active));
    items[active].scrollIntoView({ block: "nearest" });
    list.setAttribute("aria-activedescendant", items[active].id);
  }

  function choose(index: number) {
    list.hidePopover();
    if (index === select.selectedIndex)
      return;
    select.selectedIndex = index;
    select.dispatchEvent(new Event("change", { bubbles: true }));
    sync();
  }

  // Below the trigger, or above it when the list would run past the bottom of the window.
  list.addEventListener("beforetoggle", event => {
    if ((event as ToggleEvent).newState !== "open")
      return;
    const rect = trigger.getBoundingClientRect();
    list.style.minWidth = `${rect.width}px`;
    list.style.left = `${rect.left}px`;
    list.style.top = `${rect.bottom + 4}px`;
    list.style.removeProperty("--from-y");
    requestAnimationFrame(() => {
      const height = list.offsetHeight;

      if (rect.bottom + 4 + height > window.innerHeight && rect.top - 4 - height > 0) {
        list.style.top = `${rect.top - 4 - height}px`;
        list.style.setProperty("--from-y", "4px");
      }
    });
  });

  list.addEventListener("toggle", event => {
    const open = (event as ToggleEvent).newState === "open";
    trigger.setAttribute("aria-expanded", String(open));
    if (open) {
      highlight(select.selectedIndex);
      list.focus();
    } else if (document.activeElement === list || document.activeElement === document.body)
      trigger.focus();
  });

  trigger.addEventListener("keydown", event => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      list.showPopover();
    }
  });

  // Escape is handled by the popover itself.
  list.addEventListener("keydown", event => {
    const moves: Record<string, number> = {
      ArrowDown: active + 1,
      ArrowUp: active - 1,
      Home: 0,
      End: items.length - 1,
    };

    if (event.key in moves) {
      event.preventDefault();
      highlight(moves[event.key]);
    } else if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      choose(active);
    } else if (event.key === "Tab")
      list.hidePopover();
  });

  // A fixed list would drift away from its trigger when the sidebar scrolls or the window resizes.
  const close = (event: Event) => {
    if (event.target !== list && list.matches(":popover-open"))
      list.hidePopover();
  };

  document.addEventListener("scroll", close, true);
  window.addEventListener("resize", close);

  // The page toggles `disabled` on the select and may set its value directly, then sends "sync" to show it.
  new MutationObserver(sync).observe(select, { attributes: true, attributeFilter: ["disabled"] });
  select.addEventListener("change", sync);
  select.addEventListener("sync", sync);
  sync();
}
