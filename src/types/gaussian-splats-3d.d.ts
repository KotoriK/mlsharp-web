/**
 * Type declarations for @mkkellogg/gaussian-splats-3d
 */
declare module '@mkkellogg/gaussian-splats-3d' {
  export enum SceneRevealMode {
    Default = 0,
    Gradual = 1,
    Instant = 2,
  }

  export interface ViewerOptions {
    cameraUp?: [number, number, number];
    initialCameraPosition?: [number, number, number];
    initialCameraLookAt?: [number, number, number];
    rootElement?: HTMLElement;
    selfDrivenMode?: boolean;
    useBuiltInControls?: boolean;
    dynamicScene?: boolean;
    sceneRevealMode?: SceneRevealMode;
    gpuAcceleratedSort?: boolean;
    sharedMemoryForWorkers?: boolean;
  }

  export interface AddSceneOptions {
    showLoadingUI?: boolean;
    progressiveLoad?: boolean;
    position?: [number, number, number];
    rotation?: [number, number, number, number];
    scale?: [number, number, number];
  }

  export class Viewer {
    constructor(options?: ViewerOptions);
    start(): void;
    stop(): void;
    dispose(): void;
    addSplatScene(url: string, options?: AddSceneOptions): Promise<void>;
    removeSplatScenes(): Promise<void>;
    getSplatCount(): number | null;
  }
}
