"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";

export default function DetectionPage() {
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [modifiedImage, setModifiedImage] = useState<string | null>(null);
  const [personCount, setPersonCount] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [isClient, setIsClient] = useState(false); // Track if we are on client-side

  useEffect(() => {
    setIsClient(true); // Ensure rendering happens only on the client
  }, []);

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setModifiedImage(null);
    setOriginalImage(URL.createObjectURL(file)); // Preview original image
    setPersonCount(null);

    const formData = new FormData();
    formData.append("image", file);

    try {
      setLoading(true);
    //   console.log('http://localhost:8000/detection')
      const response = await fetch("http://localhost:8000/detection", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload image");
      }

      const data = await response.json();

      if (data.visualize_image) {
        setModifiedImage(`data:image/jpeg;base64,${data.visualize_image}`);
      }

      if (data.n_person !== undefined) {
        setPersonCount(data.n_person);
      }
    } catch (error) {
      console.error("Error uploading image:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-2xl font-bold mb-4 flex items-center space-x-6">
        Upload an Image
        <Link href="/" className="text-sm text-white hover:text-gray-300 hover:underline m-4">
          Home
        </Link>
        <Link href="/record" className="text-sm text-white hover:text-gray-300 hover:underline">
          View History
        </Link>
      </h1>

      <input
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        className="mb-4 p-2 border rounded"
      />

      {loading && <p className="mt-4 text-blue-500">Processing Image...</p>}

      {isClient && originalImage && modifiedImage && (
        <div className="flex flex-row items-center justify-center gap-8 mt-4">
          {/* Original Image */}
          <div>
            <h2 className="text-lg font-semibold mb-2 text-center">Original Image:</h2>
            {originalImage && (
              <Image
                src={originalImage}
                alt="Original Preview"
                width={300} // Define width
                height={200} // Define height
                className="rounded shadow-lg"
              />
            )}
          </div>

          {/* Modified Image */}
          <div>
            <h2 className="text-lg font-semibold mb-2 text-center">Visualized Image:</h2>
            {modifiedImage && (
              <Image
                src={modifiedImage}
                alt="Modified Preview"
                width={300} // Define width
                height={200} // Define height
                className="rounded shadow-lg"
              />
            )}
          </div>
        </div>
      )}

      {/* Human Detection Count */}
      {isClient && personCount !== null && (
        <h2 className="text-lg font-semibold mt-4">
          Human Detection: There {personCount === 1 ? "is" : "are"} {personCount}{" "}
          {personCount === 1 ? "person" : "people"}
        </h2>
      )}
    </div>
  );
}
