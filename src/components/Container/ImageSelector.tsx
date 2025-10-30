import React from 'react';
import { useContainerStore } from '../../store/useContainerStore';
import containerImagesData from '../../data/container-images.json';

export const ImageSelector: React.FC = () => {
  const { selectedEngineId, selectedImageId, setSelectedImageId } = useContainerStore();
  
  // Filter images by selected engine
  const availableImages = containerImagesData.images.filter(
    img => img.engine === selectedEngineId
  );
  
  const selectedImage = availableImages.find(img => img.fullImage === selectedImageId);
  
  return (
    <div className="space-y-2">
      <label htmlFor="image-select" className="block text-sm font-medium text-gray-700 dark:text-gray-300">
        Container Image
      </label>
      
      <select
        id="image-select"
        value={selectedImageId}
        onChange={(e) => setSelectedImageId(e.target.value)}
        className="w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:text-white"
      >
        {availableImages.map((image) => (
          <option key={image.fullImage} value={image.fullImage}>
            {image.fullImage} ({image.stability})
          </option>
        ))}
      </select>
      
      {selectedImage && (
        <div className="mt-2">
          {/* Description */}
          <div className="p-3 rounded-md bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
            <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
              {selectedImage.description}
            </p>
            <a
              href={`https://hub.docker.com/r/${selectedImage.repository}/tags`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center text-xs text-blue-600 dark:text-blue-400 hover:underline"
            >
              <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z" />
                <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z" />
              </svg>
              View on Docker Hub for prerequisites and details
            </a>
          </div>
        </div>
      )}
    </div>
  );
};
