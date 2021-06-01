import torch
import numpy as np
import torch.nn.functional as F

def KL(P,Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon

    divergence = np.sum(P*np.log(P/Q))
    return divergence



class WeightRegularizer():

    def __init__(self, num_classes, pretrained_model, init_param = 1, decay_weight = 0.5, beta = 0.99, number_sample = 10, mode="effective", init_size=512, class_req_perf = -1, actual_perf = 90, convex_comb = False):
        """Weight Regularizer is the regularizer based on output of a classifier. It 
        causes the GAN to avoid mode collapse

        Args:
            num_classes ([integer]): [description]
            init_param (int, optional): [description]. Defaults to 1.
            decay_weight (float, optional): [description]. Defaults to 0.5.
            beta (float, optional): [description]. Defaults to 0.99.
            mode (float, optional): [description]. If mode is effective there is effective reweights. Else inverse of class ratio.
        """
        
        self.count_class_samples = [50] * num_classes
        self.decay_weight = decay_weight
        self.beta = beta
        self.num_classes = num_classes
        self.stats = []
        self.number_to_sample = number_sample
        self.mode = mode
        self.pretrained_model = pretrained_model
        self.pred_class = [init_size] * self.num_classes
        self.convex_comb = convex_comb

        if self.mode == "effective":
            effective_num = 1.0 - np.power(self.beta, self.count_class_samples)
            weights = (1.0 - self.beta) / np.array(effective_num)
        else:
            weights = 1 / np.array(self.count_class_samples)
        
        self.weights = weights / np.sum(weights) * self.num_classes
        self.ce_loss  = torch.nn.CrossEntropyLoss()

        # For decreasing classifier accuracy for ablations
        if class_req_perf >= 0 and actual_perf >= class_req_perf:
            self.random_ratio = (actual_perf - class_req_perf) /(actual_perf - 100/num_classes)
        else:
            self.random_ratio = None
        
        print(self.random_ratio)
        

        self.i = 0

    def update(self, logger = None):
        """[summary]

        Args:
            pred_class ([list]): [List containing the predicted classes for current batch]
        """
        #pred_class_max = torch.argmax(self.pretrained_model(input_images), dim = 1)
        #pred_class = [(pred_class_max == i).sum(dim=0) for i in range(self.num_classes)]

        
        stats, kl_div = self.get_stats()
        if logger != None:
            logger.info("Mean Number of Samples %f Variance of Number of Samples %f KL Divergence of the Samples %f ."%(np.mean(stats), np.std(stats), kl_div))
        
        if self.convex_comb:
            factor = 1 - self.decay_weight
        else:
            factor = 1

        self.count_class_samples = [self.decay_weight * i for i in self.count_class_samples]
        for i in range(self.num_classes):
            self.count_class_samples[i] += (factor) * self.pred_class[i]
        print(sorted(self.count_class_samples)[-5: ]) 
        self.reset_stats()
        # Clamp the values to one for values < 1
        self.count_class_samples = [max(1, i) for i in self.count_class_samples]
        #logger.info("Updated Cumulative Samples:" + str(self.count_class_samples))

        if self.mode == "effective":
            effective_num = 1.0 - np.power(self.beta, self.count_class_samples)
            weights = (1.0 - self.beta) / np.array(effective_num)
        else:
            weights = 1 / np.array(self.count_class_samples)
        
        self.weights = weights / np.sum(weights) * self.num_classes


    def log_and_print(self, count_class_samples, writer = None):
        """[summary]

        Args:
            count_class_samples ([type]): [description]
            writer ([type], optional): [Tensorbaord Writer]. Defaults to None.
        """
        if len(self.stats) == self.number_to_sample:
            self.stats.pop(0)
        # count_class_samples = [0 for i in range(self.num_classes)]
        # for cls in pred_class:
        #     count_class_samples[cls] += 1
        print("Max and Min of class samples",max(count_class_samples), min(count_class_samples))
        self.stats.append(count_class_samples)
        
        stats = np.mean(np.array(self.stats), axis=0)
        stats = stats/np.sum(stats)
        print("Mean Number of Samples %f Standard Deviation of Number of Samples %f"%(np.mean(stats), np.std(stats)))
        
        if writer is not None:
            writer.add_scalar("12. Variance of Samples in Different Classes", np.std(stats))
            # writer.add_figure("Average of Nber of Samples of Each Class Generated in Last 10 epochs", np.asarray(stats), bins=10)

    def reset_stats(self):
        """[Reset the stats produced by classifier]
        """
        self.pred_class = [0] * self.num_classes


    def get_stats(self):
        stats = np.array(self.pred_class)
        stats = stats/np.sum(stats)
        uniform = np.mean(stats)*np.ones(len(stats))
        kl_divergence = KL(stats, uniform)
        
        return stats, kl_divergence




        
    
    def loss(self, input_images, labels=False, writer = None, target_labels = None):
        """This calculates the loss of the regularizer

        Args:
            softmax_output ([Tensor]): [Softmax output of the given batch]
        """
        temperature = 5 * 1e-1
        with torch.cuda.amp.autocast():
            if self.pretrained_model.__class__.__name__ == "ResNetV2":
                output = self.pretrained_model(F.interpolate(input_images, mode='bilinear', size=(128, 128)))
                softmax_output = torch.softmax(output, dim = 1)
            else:
                
                output =  self.pretrained_model(input_images)
                softmax_output = torch.softmax(output, dim = 1)
            
            
        #print(torch.max(softmax_output[:5], dim = 1))
        pred_class_max = torch.argmax(softmax_output, dim = 1).cpu()
        
        if self.random_ratio is not None:
            # logic for random sampling
            random_mask = torch.rand((input_images.shape[0], )).le(self.random_ratio)
            #print(random_mask, self.random_ratio)
            #print(torch.sum(random_mask).true_divide(input_images.shape[0]))
            random_labels = torch.randint(0, self.num_classes, (input_images.shape[0], ))
            pred_class_max = (~random_mask) * pred_class_max + random_mask * random_labels
        
        pred_class = [(pred_class_max == i).sum().item() for i in range(self.num_classes)]
        
        ### For debugging
        if writer is not None:
            pred_class_prob = np.array(pred_class)/np.sum(pred_class)
            count_class_prob = np.array(self.count_class_samples)/np.sum(self.count_class_samples)
            writer.add_histogram('expected', count_class_prob, self.i)
            writer.add_histogram('predicted', pred_class_prob, self.i)
            self.i += 1
            pred_class_dict = [(k,v) for v, k in enumerate(pred_class)]
            pred_class_dict = sorted(pred_class_dict, reverse=True)
            #print(pred_class_dict[:5])

        self.pred_class = [ i + j for i,j in zip(self.pred_class, pred_class)]
        

        
        sm_batch_mean = torch.from_numpy(self.weights).float().to(device = softmax_output.device)* torch.mean(softmax_output,dim=0)
        div_loss = torch.sum(sm_batch_mean*torch.log(torch.mean(softmax_output,dim=0)))


        if target_labels != None:
            div_loss += 0*self.ce_loss(output, target_labels)

        if labels:
            return div_loss, softmax_output

        return div_loss
   



 



