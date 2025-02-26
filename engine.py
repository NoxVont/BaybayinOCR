import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from corpus_parser import CorpusParser
from ultralytics import YOLO
from itertools import product

model_dir = './runs/detect/train5/weights/best.pt'
model = YOLO(model_dir)

test_dir = './dataset/test/'

def get_bbox_coords(results, class_names):
  detections = {}
  for result in results:
      bboxes = result.boxes

      for box in bboxes:
          # Key: class, value: bbox coords (xywh)
          cls = class_names[int(box.cls)]
          
          # Class correction
          cls = cls.split('_')
          cls = cls[0]
          
          detections[cls] = [int(x.item()) for x in box.xywh[0]]
          
  return detections


def find_contours(img):
  # Creating Contour Lines
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                               
  gray = cv2.GaussianBlur(gray, (25, 25), 0)
  gray = cv2.rectangle(gray, (0, 0), (gray.shape[0], gray.shape[1]) , (255, 255, 255), 10) 
  gray = cv2.bitwise_not(gray)    # Inverts the image
  
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
  p_img = cv2.dilate(thresh, kernel)    # Dilate the image to get contour lines
  
  contours, _ = cv2.findContours(p_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  return contours


def find_orientation(contours):
  # Finding all angles
  angles = []
  for contour in contours:
    
    # Using PCA
    contour = contour.reshape(-1, 2)                        # Reduce dimension of contour 
    mean = np.mean(contour, axis=0)                         # Mean of contour points
    center_contour = contour - mean                         # Centered Contour
    cov_mat = np.cov(center_contour.T)                      # Covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)   

    # The eigenvector corresponding to the largest eigenvalue is the principal axis
    max_index = np.argmax(eigenvalues)
    principal_axis = eigenvectors[:, max_index]

    # The angle is between the principal axis and the horizontal axis
    angle = np.arctan2(principal_axis[1], principal_axis[0]) * 180 / np.pi  
    angle = round(angle / 90) * 90
    angles.append(angle)
  
  # Text Orientation
  orientation = 0       #   0 = horizontal, 1 = vertical
  orient_angle = 0
  min_diff = float('inf')
  
  for angle in angles:
    for target_angle in [0, 90, 180, -90]:
      diff = abs(angle - target_angle)
      if diff < min_diff:
        min_diff = diff
        orient_angle = target_angle
  return 0 if orient_angle in [0, 180] else 1
  

def group_to_words(detections, orientation):
  detection_list = [(cls, box[0], box[1]) for cls, box in detections.items()]
  detection_list = sorted(detection_list, key=lambda x: x[orientation + 1])
  
  
  # Find the mean of the opposing axis to find the threshold (average height of detections) 
  h_dets = [box[3] for box in detections.values()]
  v_dets = [box[2] for box in detections.values()]
  
  # Threshold distance, if the next character is more than one character away, its not part of the word
  if orientation == 0:
      h_threshold = np.mean(h_dets) * 2.75
      v_threshold = np.mean(v_dets)
  elif orientation == 1:
      h_threshold = np.mean(h_dets) 
      v_threshold = np.mean(v_dets) * 2
  
  # Word Grouping
  words = []
  current = []

  for i in range(len(detection_list) - 1):
      h_distance = abs(detection_list[i+1][1] - detection_list[i][1])   
      v_distance = abs(detection_list[i+1][2] - detection_list[i][2])
      
      # If the next character is within the threshold for opposite axis, it is accepted as part of the word
      if orientation == 0:
        if v_distance < v_threshold:
            current.append(detection_list[i][0])  # Append the class to the current
            
            # If the next character is beyond threshold, word ends 
            if h_distance > h_threshold:
                words.append(''.join(current))
                current = []        
            
      elif orientation == 1:
        if h_distance < h_threshold:
            current.append(detection_list[i][0])  # Append the class to the current
            
            # If the next character is beyond threshold, word ends 
            if v_distance > v_threshold:
                words.append(''.join(current))
                current = []    
      
  # Last word
  current.append(detection_list[-1][0])
  words.append(''.join(current))
  
  return words
      
      
def correct_words(word_group):
  # Load corpus
  cp = CorpusParser('./corpora/tagalog-corpus.txt')

  corrected_words = []
  for possible_words in word_group:
      # Check each sublist of potential words
      correct_words = []
      for word in possible_words:
          if cp.check_word(word):
              correct_words.append(word)
          else:
              word = cp.correct_word(word)
              if cp.check_word(word):
                  correct_words.append(word)
      
      corrected_words.append(correct_words)
  
  # Structure:
  # [ [], [], ... ]
  return corrected_words


def permutate_words(word_group):
  # Load corpus
  cp = CorpusParser('./corpora/tagalog-corpus.txt')
  permuted_words = []
  
  for word in word_group:
      possible_words = []
      possible_words.append(word)
      # D and R interchangeability, change D to R if applicable
      # Only applicable if D is not the first or last letter and if preceding char before D is a vowel
      if 'd' in word:
          for i in range(1, len(word)-1):
              if word[i] == 'd' and word[i-1] in 'aeiou':
                  permuted = word[:i] + 'r' + word[i+1:]
                  possible_words.append(permuted)
                  break
                      

      # Vowel interchangeability   
      total_perms = []
      for w in possible_words:
          permuted_chars = []
          for char in w:
              if char == 'e':
                  permuted_chars.append(['e', 'i'])
              elif char == 'i':
                  permuted_chars.append(['i', 'e'])
              elif char == 'o':
                  permuted_chars.append(['o', 'u'])
              elif char == 'u':
                  permuted_chars.append(['u', 'o'])
              else: 
                  permuted_chars.append([char])
                  
          # Cartesian product for all interchangeable vowels     
          permutations = [''.join(p) for p in product(*permuted_chars)]
          total_perms.append(permutations)
          
      for permutations in total_perms:
          for perm in permutations:
              possible_words.append(perm) 
      
      possible_words = list(set(possible_words))  # Remove duplicates
      permuted_words.append(possible_words)
      
 
  # Final corpus check
  filter = []
  for possible_words in permuted_words:
      # Possible words is a sublist of potential words
      filtered_words = []
      for word in possible_words:
          if cp.check_word(word):
              filtered_words.append(word)
      filter.append(filtered_words)
      
  permuted_words = filter
  return permuted_words  
            

def read_image(file_path, b_model):
  img = cv2.imread(file_path, cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  results = b_model(img)
  class_names = b_model.names
  detections = get_bbox_coords(results, class_names)

  contours = find_contours(img)
  orientation = find_orientation(contours)
  word_group = group_to_words(detections, orientation)
  possible_words = permutate_words(word_group) # After Permutation, the word group becomes 2D
  possible_words = correct_words(possible_words)

  predicted_word = ""
  for words in possible_words:
      # Words with 2 or more possibilites have an OR in prediction
      if len(words) > 1:
        predicted_word += ' or '.join(words)
      elif len(words) == 1: 
        predicted_word += ' '.join(words) 
  print(f"Detections: {[key for key in detections.keys()]}")
  print(f"Predicted Word: {predicted_word}")
  
  # Display Results
  fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
  # Original Image
  ax[0].imshow(img)
  ax[0].set_title('Original Image')
  ax[0].axis('off')

  # Image with Bboxes and orientation
  ax[1].imshow(results[0].plot())
  ax[1].set_title('Processed Image')
  ax[1].axis('off')
  
  plt.tight_layout()
  plt.show()
  
  return predicted_word
    
    
# Purely for testing purposes
if __name__ == "__main__":
    # Evaluation Metrics
    char_detections = {}
    word_accuracy = 0.0
    word_error_rate = 1.0
    
    word_total_preds = 0
    word_correct_preds = 0
    
    handwritten_pred = 0.0
    typewritten_pred = 0.0
    
    for f in os.listdir(test_dir):
      # Get true label
      id = int(f.split("_")[0])
      true_word = f.split("_")[1].split('.')[0]
      
      img = cv2.imread(os.path.join(test_dir, f), cv2.IMREAD_COLOR)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      word_total_preds += 1
      print(f"Test Image:   {f}")

      # Preprocess, then detect, then display in subplots
      # preprocessed_img = preprocess_image(img)
      results = model(img)
      class_names = model.names
      
      
      # Get bbox coordinates
      detections = get_bbox_coords(results)

      
      blank_img = np.full_like(img, 255) # For displaying contours and principal axis only
      
      
      contours = find_contours(img)
      orientation = find_orientation(contours)
      word_group = group_to_words(detections, orientation)
      possible_words = permutate_words(word_group) # After Permutation, the word group becomes 2D
      possible_words = correct_words(possible_words)

      predicted_word = ""
      for words in possible_words:
          # Words with 2 or more possibilites have an OR in prediction
          if len(words) > 1:
            predicted_word += ' or '.join(words)
          elif len(words) == 1: 
            predicted_word += ' '.join(words) 
      
      """
      # Display Results
      fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
      # Original Image
      ax[0].imshow(img)
      ax[0].set_title('Original Image')
      ax[0].axis('off')

      # Image with Bboxes and orientation
      ax[1].imshow(results[0].plot())
      # ax[1].imshow(p_img, cmap='gray')
      ax[1].set_title('Processed Image')
      ax[1].axis('off')
      

      # Contour Points
      ax[2].imshow(blank_img)
      ax[2].set_title('Contour Points')
      ax[2].axis('off')
      
      plt.tight_layout()
      plt.show()
      """
      
      print(f"\nPredicted Word: {predicted_word}")
      print(f"True Word:      {true_word}")
      
      for words in possible_words:
          if true_word in words:
              word_correct_preds += 1
              
              if id % 4 in [1, 2]: 
                  handwritten_pred += 1
              elif id % 4 in [0, 3]:
                  typewritten_pred += 1
      
      word_accuracy = (word_correct_preds / word_total_preds) 
      print(f"Word Accuracy:  {word_accuracy}")

      
      for det in detections.keys():
        if det in char_detections:
          char_detections[det] += 1
        else:
          char_detections[det] = 1
      
    handwritten_accuracy = handwritten_pred / (word_total_preds / 2)
    typewritten_accuracy = typewritten_pred / (word_total_preds / 2)
      
    print("\nFinal Results")
    print(f"Word Accuracy:  {word_accuracy}")
    print(f"Handwritten Accuracy:  {handwritten_accuracy}")
    print(f"Typewritten Accuracy:  {typewritten_accuracy}")
    print("\nCharacter Detections:")
    for char, dets in char_detections.items():
        print(f"{char}:    {dets}")
