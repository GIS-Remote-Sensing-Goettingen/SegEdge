from transformers import pipeline
from transformers.image_utils import load_image
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Example labeled images (cat and dog)
cat_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
dog_url = "https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg"

cat_image = load_image(cat_url)
dog_image = load_image(dog_url)

feature_extractor = pipeline(
    model="facebook/dinov3-vitl16-pretrain-sat493m",
    task="image-feature-extraction",
)

cat_feat = feature_extractor(cat_image)
dog_feat = feature_extractor(dog_image)

cat_feat = np.array(cat_feat).reshape(1, -1)
dog_feat = np.array(dog_feat).reshape(1, -1)

X_train = np.vstack([cat_feat, dog_feat])
y_train = np.array([0, 1])  # 0: cat, 1: dog

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#test and stuff
test_url = "https://cdn2.thecatapi.com/images/MTY3ODIyMQ.jpg"
test_image = load_image(test_url)
test_feat = feature_extractor(test_image)
test_feat = np.array(test_feat).reshape(1, -1)
pred = knn.predict(test_feat)
label_map = {0: "cat", 1: "dog"}
print(f"Predicted class: {label_map[pred[0]]}")
