import { useCallback, useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  previewUrl: string | null;
}

export default function ImageUpload({ onFile, previewUrl }: Props) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files || files.length === 0) return;
      const file = files[0];
      if (!file.type.startsWith("image/")) return;
      onFile(file);
    },
    [onFile]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  return (
    <div
      className={`relative rounded-2xl overflow-hidden transition-all duration-200 cursor-pointer
        ${dragging ? "ring-2 ring-purple-500 ring-offset-2 ring-offset-[#0a0a14]" : ""}
        ${previewUrl ? "aspect-video" : "aspect-[4/3]"}
        glass hover:border-purple-500/40`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />

      {previewUrl ? (
        <>
          <img
            src={previewUrl}
            alt="Uploaded"
            className="w-full h-full object-contain bg-[#0d0d1f]"
          />
          <div className="absolute inset-0 flex items-end justify-center pb-3 opacity-0 hover:opacity-100 transition-opacity">
            <span className="glass text-xs text-slate-300 px-3 py-1.5 rounded-full">
              Click to change image
            </span>
          </div>
        </>
      ) : (
        <div className="flex flex-col items-center justify-center h-full gap-4 p-8 text-center select-none">
          <div className="w-16 h-16 rounded-2xl flex items-center justify-center bg-gradient-to-br from-purple-600/30 to-blue-600/30 border border-purple-500/20">
            <svg className="w-8 h-8 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <div>
            <p className="text-slate-200 font-medium mb-1">
              {dragging ? "Drop it here!" : "Drop an image or click to browse"}
            </p>
            <p className="text-slate-500 text-sm">PNG · JPG · WebP · BMP</p>
          </div>
        </div>
      )}
    </div>
  );
}
