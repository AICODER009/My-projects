# Neural Network Classification 
classification.ipynb - application of linearity/non-linearity(RELU) principles on color detection problems

**Note: it's a brief overview what had been done. Please for the detailed information follow the Collab Version.**

![image](https://github.com/AICODER009/Pytorch_projects/assets/133597851/643cfe3f-e22c-4726-86f5-87f1378f98d7)
A visual example of what a similar classificiation neural network to the one we've just built (using ReLU activation) looks like.

## **Evaluating a model trained with non-linear activation function**
**Make predictions**
``` ruby
model_3.eval()
with torch.inference_mode():
    y_preds = torch.round(torch.sigmoid(model_3(X_test))).squeeze()
y_preds[:10], y[:10] # want preds in same format as truth labels
```
```ruby
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_1, X_train, y_train) # model_1 = no non-linearity
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test) # model_3 = has non-linearity
```
![image](https://github.com/AICODER009/Pytorch_projects/assets/133597851/e4c4674e-5190-4aff-ac9b-b41954c90e80)

Hint: Potentially a few tricks can be applied in order to improve the test accuracy of the model.
## **Creating mutli-class classification data**

1.Create some multi-class data with make_blobs();

2.Turn the data into tensors (the default of make_blobs() is to use NumPy arrays);

3.Split the data into training and test sets using train_test_split();

4.Visualize the data.

![image](https://github.com/AICODER009/Pytorch_projects/assets/133597851/fa66b544-5f0a-406b-b9d0-098a6e783941)

## Making and evaluating predictions with a PyTorch multi-class model
``` ruby
model_4.eval()
with torch.inference_mode():
    y_logits = model_4(X_blob_test)
```
It's possible to skip the torch.softmax() function and go straight from predicted logits -> predicted labels by calling torch.argmax() directly on the logits.

For example, y_preds = torch.argmax(y_logits, dim=1), this saves a computation step (no torch.softmax()) but results in no prediction probabilities being available to use.

``` ruby
y_pred_probs = torch.softmax(y_logits, dim=1)

# Turn prediction probabilities into prediction labels
y_preds = y_pred_probs.argmax(dim=1)

```
And by visualising we get test accuracy of 99.5% with correctly classified colours:
![image](https://github.com/AICODER009/Pytorch_projects/assets/133597851/66071a9f-f5c7-44c0-a531-83c3f8d03fc0)
