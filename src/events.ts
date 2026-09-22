export type Listener<T> = (payload: T) => void;

/** Typed event emitter: `on` returns the function that removes the listener. */
export class EventEmitter<Events extends object> {
  private listeners: { [K in keyof Events]?: Set<Listener<Events[K]>> } = {};

  on<K extends keyof Events>(type: K, listener: Listener<Events[K]>): () => void {
    (this.listeners[type] ??= new Set()).add(listener);

    return () => this.off(type, listener);
  }

  /** Listens for the next event of that type only. */
  once<K extends keyof Events>(type: K, listener: Listener<Events[K]>): () => void {
    const off = this.on(type, payload => {
      off();
      listener(payload);
    });

    return off;
  }

  off<K extends keyof Events>(type: K, listener: Listener<Events[K]>) {
    this.listeners[type]?.delete(listener);
  }

  /** Removes the listeners of one event type, or all of them. */
  removeAllListeners(type?: keyof Events) {
    if (type === undefined)
      this.listeners = {};
    else
      delete this.listeners[type];
  }

  protected emit<K extends keyof Events>(type: K, ...args: Events[K] extends void ? [] : [Events[K]]) {
    const listeners = this.listeners[type];
    if (!listeners)
      return;
    // A listener removed while the event is delivered, by itself or by another one, is not called.
    for (const listener of [...listeners])
      if (listeners.has(listener))
        listener(args[0] as Events[K]);
  }
}
