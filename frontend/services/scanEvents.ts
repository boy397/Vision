/**
 * Lightweight event emitter for scan commands.
 * Voice tab emits → Scanner tab listens.
 */

type ScanCommand = 'scan' | 'scan_continue' | 'scan_stop';
type Listener = (command: ScanCommand) => void;

class ScanEventEmitter {
    private listeners: Set<Listener> = new Set();

    on(listener: Listener) {
        this.listeners.add(listener);
        return () => this.listeners.delete(listener);
    }

    emit(command: ScanCommand) {
        this.listeners.forEach((fn) => fn(command));
    }
}

export type { ScanCommand };
export const scanEvents = new ScanEventEmitter();
export default scanEvents;
