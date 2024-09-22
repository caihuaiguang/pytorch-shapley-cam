import numpy as np
import torch
import torchvision


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


class ClassifierOutputSoftmaxTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return torch.softmax(model_output, dim=-1)[self.category]
        return torch.softmax(model_output, dim=-1)[:, self.category]


class ClassifierOutputResidualSoftmaxTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category] + torch.softmax(model_output, dim=-1)[self.category]
        return model_output[:, self.category] + torch.softmax(model_output, dim=-1)[:, self.category]

class ClassifierOutputLnSoftmaxTarget:
    def __init__(self, category):
        self.category = category  # This is the target category

    def __call__(self, model_output):
        """
        Calculate the Cross-Entropy Loss for the given target category.
        
        Parameters:
        - model_output (torch.Tensor): The raw output (logits) from the model.
        
        Returns:
        - torch.Tensor: Cross-Entropy Loss value for the specified category.
        """
        # Check the dimensionality of the model output
        if len(model_output.shape) == 1:
            # Convert the target category to a tensor
            target = torch.tensor([self.category], device=model_output.device)
            # Reshape model_output to match the expected input for cross-entropy loss
            model_output = model_output.unsqueeze(0)
            # Calculate cross-entropy loss
            return - torch.nn.functional.cross_entropy(model_output, target)
        else:
            # For batch-wise model output
            target = torch.tensor([self.category] * model_output.shape[0], device=model_output.device)
            return - torch.nn.functional.cross_entropy(model_output, target)


class ClassifierOutputExclusiveLnSoftmaxTarget:
    def __init__(self, category):
        self.category = category

    # def __call__(self, model_output):
    #     Probs = torch.softmax(model_output, dim=-1)
    #     LnProbs = torch.log(Probs)
    #     if len(model_output.shape) == 1:
    #         # Apply softmax followed by natural logarithm
    #         return LnProbs[self.category] - torch.mean(LnProbs)
    #     # For multi-dimensional output
    #     return LnProbs[:, self.category] - torch.mean(LnProbs, dim=-1)

    
    def __call__(self, model_output):
        Probs = torch.softmax(model_output, dim=-1) 
        if len(model_output.shape) == 1:
            # Apply softmax followed by natural logarithm
            return 2*Probs[self.category] - torch.sum(Probs)
        # For multi-dimensional output
        return 2*Probs[:, self.category] - torch.sum(Probs, dim=-1) 
    
    
    
class ClassifierOutputEntropy:
    def __init__(self, category):
        self.category = category

    # def __call__(self, model_output): 
    #     probabilities = torch.softmax(model_output, dim=-1)
    #     # minimize entropy, so max (-entropy)
    #     entropy = torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=-1)
        
    #     return entropy
    
    def __call__(self, model_output):
        # if len(model_output.shape) == 1:
        #     # Apply softmax followed by natural logarithm
        #     return torch.log(torch.softmax(model_output/2, dim=-1)[self.category])
        # # For multi-dimensional output
        # return torch.log(torch.softmax(model_output/2, dim=-1)[:, self.category])

        # if len(model_output.shape) == 1:
        #     # Apply softmax followed by natural logarithm
        #     return torch.log(torch.softmax(model_output/0.5, dim=-1)[self.category])
        # # For multi-dimensional output
        # return torch.log(torch.softmax(model_output/0.5, dim=-1)[:, self.category])

        epsilon = 1e-12 
        if len(model_output.shape) == 1:
            # Apply softmax followed by natural logarithm
            return - torch.log(1-torch.softmax(model_output, dim=-1)[self.category]+ epsilon) 
        # For multi-dimensional output
        return - torch.log(1 - torch.softmax(model_output, dim=-1)[:, self.category]+ epsilon)
        # epsilon = 1e-12 
        # if len(model_output.shape) == 1:
        #     # Apply softmax followed by natural logarithm
        #     return - torch.log(1-torch.softmax(model_output, dim=-1)[self.category]+ epsilon)
        # # For multi-dimensional output
        # return - torch.log(1 - torch.softmax(model_output, dim=-1)[:, self.category]+ epsilon)

class ClassifierOutputReST_original:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            # Single sample case
            with torch.no_grad():
                s = torch.softmax(model_output, dim=-1)  # s is the softmax result
                s_c = s[self.category]  # s_c is the score of class c in s

            o_c = model_output[self.category]  # o_c is the score of class c in model_output
            dot_product = (s * model_output).sum()  # Calculate <s, o> as a dot product

            # Calculate the expression (1 + s_c) * o_c - <s, o>
            result = (1 + s_c) * o_c - dot_product
        
        elif len(model_output.shape) == 2:
            # Batch of samples case
            with torch.no_grad():
                s = torch.softmax(model_output, dim=-1)  # s is the softmax result for each sample
                s_c = s[:, self.category]  # s_c is the score of class c in s for each sample

            o_c = model_output[:, self.category]  # o_c is the score of class c in model_output for each sample
            dot_product = (s * model_output).sum(dim=-1)  # Calculate <s, o> as a dot product along sample dimension
            
            # Calculate the expression (1 + s_c) * o_c - <s, o> for each sample
            result = (1 + s_c) * o_c - dot_product
        
        else:
            raise ValueError("The dimension of model_output must be 1 or 2.")
        
        return result


    # def __call__(self, model_output):
    #     epsilon = 1e-14 
    #     if len(model_output.shape) == 1:
    #         # Single sample case
    #         with torch.no_grad():
    #             s = torch.softmax(model_output, dim=-1)  # s is the softmax result
    #             s_c = s[self.category]  # s_c is the score of class c in s

    #         o_c = model_output[self.category]  # o_c is the score of class c in model_output
    #         dot_product = (s * model_output).sum()  # Calculate <s, o> as a dot product

    #         # Calculate the expression (1 + s_c) * o_c - <s, o>
    #         result = o_c - (dot_product-s_c * o_c+ epsilon)/(1-s_c+ epsilon)
    #         # result = o_c 
        
    #     elif len(model_output.shape) == 2:
    #         # Batch of samples case
    #         with torch.no_grad():
    #             s = torch.softmax(model_output, dim=-1)  # s is the softmax result for each sample
    #             s_c = s[:, self.category]  # s_c is the score of class c in s for each sample

    #         o_c = model_output[:, self.category]  # o_c is the score of class c in model_output for each sample
    #         dot_product = (s * model_output).sum(dim=-1)  # Calculate <s, o> as a dot product along sample dimension
            
    #         # Calculate the expression (1 + s_c) * o_c - <s, o> for each sample
    #         result = o_c - (dot_product-s_c * o_c+ epsilon)/(1-s_c+ epsilon)
        
    #     else:
    #         raise ValueError("The dimension of model_output must be 1 or 2.")
        
    #     return result


    # def __call__(self, model_output):
    #     if len(model_output.shape) == 1:
    #         # Single sample case
    #         with torch.no_grad():
    #             s = torch.softmax(model_output, dim=-1)  # s is the softmax result 
    #         o_c = model_output[self.category]  # o_c is the score of class c in model_output
    #         dot_product = (s * model_output).sum()  # Calculate <s, o> as a dot product

    #         # Calculate the expression (1 + s_c) * o_c - <s, o>
    #         result =  o_c - dot_product
        
    #     elif len(model_output.shape) == 2:
    #         # Batch of samples case
    #         with torch.no_grad():
    #             s = torch.softmax(model_output, dim=-1)  # s is the softmax result for each sample

    #         o_c = model_output[:, self.category]  # o_c is the score of class c in model_output for each sample
    #         dot_product = (s * model_output).sum(dim=-1)  # Calculate <s, o> as a dot product along sample dimension
            
    #         # Calculate the expression (1 + s_c) * o_c - <s, o> for each sample
    #         result = o_c - dot_product
        
    #     else:
    #         raise ValueError("The dimension of model_output must be 1 or 2.")
        
    #     return result


class ClassifierOutputReST:
    def __init__(self, category, temperature=0.9, epsilon=0):
        self.category = category
    def __call__(self, model_output): 
        """
        Computes the GIST loss for the given model output.

        Args:
            model_output (torch.Tensor): The model output logits.

        Returns:
            torch.Tensor: The computed GIST loss.
        """
        # Check the dimensionality of the model output
        if len(model_output.shape) == 1:
            # Convert the target category to a tensor
            target = torch.tensor([self.category], device=model_output.device)
            # Reshape model_output to match the expected input for cross-entropy loss
            # score = model_output[self.category]
            model_output = model_output.unsqueeze(0)
            # Calculate cross-entropy loss
            return model_output[0][self.category] - torch.nn.functional.cross_entropy(model_output, target)
        else:
            # For batch-wise model output
            target = torch.tensor([self.category] * model_output.shape[0], device=model_output.device)
            return model_output[:,self.category]- torch.nn.functional.cross_entropy(model_output, target)


class ClassifierOutputReST_2:
    def __init__(self, category, temperature=0.9, epsilon=0):
        self.category = category
    def __call__(self, model_output): 
        """
        Computes the GIST loss for the given model output.

        Args:
            model_output (torch.Tensor): The model output logits.

        Returns:
            torch.Tensor: The computed GIST loss.
        """
        # Check the dimensionality of the model output
        if len(model_output.shape) == 1:
            # Convert the target category to a tensor
            target = torch.tensor([self.category], device=model_output.device)
            # Reshape model_output to match the expected input for cross-entropy loss
            # score = model_output[self.category]
            score = model_output[self.category]
            model_output = model_output.unsqueeze(0)
            # Calculate cross-entropy loss
            utility = score - torch.nn.functional.cross_entropy(model_output, target)
            return utility
        else:
            # For batch-wise model output
            target = torch.tensor(self.category, device=model_output.device)
            utility = model_output[:,self.category].mean() - torch.nn.functional.cross_entropy(model_output, target)
            print(utility)
            return utility
class BinaryClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if self.category == 1:
            sign = 1
        else:
            sign = -1
        return model_output * sign


class SoftmaxOutputTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        return torch.softmax(model_output, dim=-1)


class RawScoresOutputTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        return model_output


class SemanticSegmentationTarget:
    """ Gets a binary spatial mask and a category,
        And return the sum of the category scores,
        of the pixels in the mask. """

    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        if torch.backends.mps.is_available():
            self.mask = self.mask.to("mps")

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


class FasterRCNNBoxScoreTarget:
    """ For every original detected bounding box specified in "bounding boxes",
        assign a score on how the current bounding boxes match it,
            1. In IOU
            2. In the classification score.
        If there is not a large enough overlap, or the category changed,
        assign a score of 0.

        The total score is the sum of all the box scores.
    """

    def __init__(self, labels, bounding_boxes, iou_threshold=0.5):
        self.labels = labels
        self.bounding_boxes = bounding_boxes
        self.iou_threshold = iou_threshold

    def __call__(self, model_outputs):
        output = torch.Tensor([0])
        if torch.cuda.is_available():
            output = output.cuda()
        elif torch.backends.mps.is_available():
            output = output.to("mps")

        if len(model_outputs["boxes"]) == 0:
            return output

        for box, label in zip(self.bounding_boxes, self.labels):
            box = torch.Tensor(box[None, :])
            if torch.cuda.is_available():
                box = box.cuda()
            elif torch.backends.mps.is_available():
                box = box.to("mps")

            ious = torchvision.ops.box_iou(box, model_outputs["boxes"])
            index = ious.argmax()
            if ious[0, index] > self.iou_threshold and model_outputs["labels"][index] == label:
                score = ious[0, index] + model_outputs["scores"][index]
                output = output + score
        return output
