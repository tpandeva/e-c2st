from trainer import Trainer
import numpy as np
import torch
from scipy.optimize import minimize

class TrainerC2ST(Trainer):

    def __init__(self, cfg, net, tau1, tau2, datagen, device, data_seed):
        """
        Initializes the TrainerC2ST object by extending the Trainer class.

        Args:
        (same as Trainer class)
        """
        super().__init__(cfg, net, tau1, tau2, datagen, device, data_seed)
        self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
        self.lambda_m = cfg.lambda_null

    def e_c2st(self, y, logits):
        """
        Evaluate the E-C2ST for given targets and logits.

        Args:
        - y (torch.Tensor): Ground truth labels.
        - logits (torch.Tensor): Model outputs before activation.

        Returns:
        - torch.Tensor: Evaluated E-value.
        """
        # H0: Empirical frequencies
        emp_freq_class0 = 1 - (y[y == 1]).sum() / y.shape[0]
        emp_freq_class1 = (y[y == 1]).sum() / y.shape[0]

        # H1: Probabilities under empirical model (using train data)
        f = torch.nn.Softmax(dim=1)
        prob = f(logits).detach()
        pred_prob_class0 = prob[:, 0]
        pred_prob_class1 = prob[:, 1]
        p = self.lambda_m
        ratio1 = pred_prob_class1.detach() / emp_freq_class1.detach()
        ratio0 = pred_prob_class0.detach() / emp_freq_class0.detach()
        log_eval = torch.sum(
            y * torch.log(p + (1 - p) * ratio1.detach()) + (1 - y) * torch.log(p + (1 - p) * ratio0.detach()))
        eval = torch.exp(log_eval.detach())

        def fun(p):
            log_eval_loss = -np.sum(y.detach().cpu().numpy() * np.log(p + (1 - p) * ratio1.detach().cpu().numpy()) + (
                        1 - y.detach().cpu().numpy()) * np.log(p + (1 - p) * ratio0.detach().cpu().numpy()))
            return log_eval_loss

        self.lambda_m = minimize(fun, p, method="L-BFGS-B", bounds=[(0.01, 0.99)]).x[0]

        return eval

    def first_k_unique_permutations(self, n, k):
        """
        Generate the first k unique permutations of range(n).

        Args:
        - n (int): Size of the set.
        - k (int): Number of unique permutations required.

        Returns:
        - list: List of k unique permutations.
        """
        if np.log(k) > n * (np.log(n) - 1) + 0.5 * (np.log(2 * np.pi * n)):
            k = n
        unique_perms = set()
        while len(unique_perms) < k:
            unique_perms.add(tuple(np.random.choice(n, n, replace=False)))
        return list(unique_perms), k

    def s_c2st(self, y, logits, n_per=500):
        """
        Evaluate the permutation-based Two-Sample Test (TST) for given labels and logits.

        Args:
        - y (torch.Tensor): Ground truth labels.
        - logits (torch.Tensor): Model outputs before activation.
        - n_per (int, optional): Number of permutations. Default is 500.

        Returns:
        - tuple: p-value and accuracy.
        """
        y_hat = torch.argmax(logits, dim=1)
        n = y.shape[0]
        accuracy = torch.sum(y == y_hat) / n
        stats = np.zeros(n_per)
        permutations, n_per = self.first_k_unique_permutations(n, n_per)
        for r in range(n_per):
            ind = np.asarray(permutations[r])
            y_perm = y.clone()[ind]
            stats[r] = torch.sum(y_perm == y_hat) / y.shape[0]
        sorted_stats = np.sort(stats)
        p_val = (np.sum(sorted_stats >= accuracy.item())+1) / (n_per+1)

        return p_val, accuracy

    def m_c2st(self, y, logits, n_per=500):
        """
        Evaluate the permutation-based Two-Sample Test (TST) for given labels and logits.

        Args:
        - y (torch.Tensor): Ground truth labels.
        - logits (torch.Tensor): Model outputs before activation.
        - n_per (int, optional): Number of permutations. Default is 500.

        Returns:
        - tuple: p-value.
        """
        logit = logits.detach().cpu().numpy()
        y_ = y.detach().cpu().numpy()
        n= y.shape[0]
        true_stat = np.linalg.norm(logit[y_ == 1,:].mean(0) - logit[y_ == 0,:].mean(0))**2
        stats = np.zeros(n_per)
        permutations, n_per = self.first_k_unique_permutations(n, n_per)
        for r in range(n_per):
            ind = np.asarray(permutations[r])
            logit_perm = logit.copy()[ind,:]
            stats[r] =  np.linalg.norm(logit_perm[y_ == 1,:].mean(0) - logit_perm[y_ == 0,:].mean(0))**2
        sorted_stats = np.sort(stats)
        p_val = (np.sum(sorted_stats >= true_stat.item())+1) / (n_per+1)

        return p_val

    def l_c2st(self, y, logits, n_per=500):
        """
        Evaluate the permutation-based Two-Sample Test (TST) for given labels and logits.

        Args:
        - y (torch.Tensor): Ground truth labels.
        - logits (torch.Tensor): Model outputs before activation.
        - n_per (int, optional): Number of permutations. Default is 100.

        Returns:
        - tuple: p-value and accuracy.
        """
        y_hat = torch.argmax(logits, dim=1)
        logit = logits[:,1] - logits[:,0]
        n= y.shape[0]
        true_stat = logit[y == 1].mean() - logit[y == 0].mean()
        stats = np.zeros(n_per)
        permutations, n_per = self.first_k_unique_permutations(n, n_per)
        for r in range(n_per):
            ind = np.asarray(permutations[r])
            logit_perm = logit.clone()[ind]
            stats[r] = logit_perm[y == 1].mean() - logit_perm[y == 0].mean()
        sorted_stats = np.sort(stats)
        p_val = (np.sum(sorted_stats >= true_stat.item())+1) / (n_per+1)

        return p_val

    def train_evaluate_epoch(self, loader, mode="train"):
        """
        Train/Evaluate the model for one epoch using the C2ST approach.

        Args:
        - loader (DataLoader): DataLoader object to iterate through data.
        - mode (str): Either "train" or "test". Determines how to run the model.

        Returns:
        - tuple: Aggregated loss and E-value using ONS betting strategy for the current epoch.
        """
        aggregated_loss = 0
        e_val, tb_val_ons = 1, 1
        num_samples = len(loader.dataset)
        for i, (z, tau_z) in enumerate(loader):
            z = z.to(self.device)
            tau_z = tau_z.to(self.device)
            if mode == "train":
                self.net = self.net.train()
                out1 = self.net(z)
                out2 = self.net(tau_z)
            else:
                self.net = self.net.eval()
                out1 = self.net(z)
                out2 = self.net(tau_z)
            out = torch.concat((out1, out2))
            labels = torch.concat((torch.ones((z.shape[0], 1)), torch.zeros((z.shape[0], 1)))).squeeze(1).long().to(
                self.device)
            loss = self.loss(out, labels)
            aggregated_loss += loss

            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Compute the C2ST and other evaluation metrics
            if mode == "test":
                e_val *= self.e_c2st(labels, out.detach())
                p_val, acc = self.s_c2st(labels, out.detach())
                l_val = self.l_c2st(labels, out.detach())
                m_val = self.m_c2st(labels, out.detach())


                self.log(
                    {f"{mode}_e-value": e_val.item(),
                     f"{mode}_p-value-lc2st": l_val.item(),
                     f"{mode}_p-value-sc2st": p_val.item(),
                     f"{mode}_p-value-mc2st": m_val,
                     })

            self.log({
                f"{mode}_loss": aggregated_loss.item() / num_samples
            })

        return aggregated_loss / num_samples, e_val

