import os

import torch

from utils import progress_bar, ReadDataBase

from deep_utils import * # CudaPytorchunique, CountAllWeights, CountZeroWeights, SumAllWeights

class LearningProcess:
    def __init__(self, DATA, criterion, optimizer, schedule_factor=0.1, gpus=None):
        self.trainloader, self.testloader = ReadDataBase(DATA)
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedule_factor = schedule_factor
        self.gpus = gpus
        self.best_acc = 0
    def scheduler(self):
        self.optimizer.param_groups[0]['lr'] = self.optimizer.param_groups[0]['lr']*self.schedule_factor
        self.printlr()
    def lr(self, lr):
        self.optimizer.param_groups[0]['lr'] = lr
    def InsertNet(self, net):
        self.net = net
        self.optimizer.param_groups[0]['params'] = list(self.net.parameters())  # I suffered much to find this thing =(
    def printlr(self):
        print(colored("Current lr={0}".format(self.optimizer.param_groups[0]['lr']), "blue"))
    def test(self, epoch, namemodel=None):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                total += targets.size(0)
                correct += outputs.max(1)[1].eq(targets.data).cpu().sum().item()

                progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        weights_values = SumAllWeights(self.net)
        if np.isnan(weights_values):
            print(  colored('========================\n========================\n','red'), 
                    colored(weights_values,'red'),'\n',
                    colored('========================\n========================\n','red'), )

        # Save checkpoint.
        acc = 100.*correct/total
        if acc >= self.best_acc:
            if namemodel:
                print(colored('Saving in '+ str(namemodel),'green'))
                state = {}
                try: # Learn how to read modules correctly
                    if self.gpus:  state['net'] = self.net.base_model.module.state_dict()
                    else:  state['net'] = self.net.base_model.state_dict()
                except:
                    if self.gpus:  state['net'] = self.net.module.state_dict()
                    else:  state['net'] = self.net.state_dict()
                state['acc'] = acc
                state['epoch'] = epoch

                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/'+namemodel+'.t7')
            self.best_acc = acc
        return test_loss/(batch_idx+1), 100.*correct/total

    # Training
    def train(self, epoch,stage):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()

            all_layers = self.net.all_layers
            if stage=='Pruning':
                # zero-out all the gradients corresponding to the pruned connections
                for cnt, par in enumerate(self.net.parameters()):
                    if 'weight' in all_layers[cnt]:
                        pruned_inds = (par!=0).data
                        # if batch_idx%100==0:
                        #     print( name, "  ", all_layers[cnt], "  ", par.abs().sum().item(), "  ",  (par.grad- Variable(pruned_inds.float() * par.grad.data)).sum().item()  )
                        par.grad.data = pruned_inds.float() * par.grad.data
            elif stage=='WeightSharing':
                # zero-out all the gradients corresponding to the pruned connections and use only shared weights
                for cnt, par in enumerate(self.net.parameters()):
                    if 'weight' in all_layers[cnt]:
                        par.grad = Variable((par!=0.).float() * par.grad.data)
                        # w = CudaPytorchunique(p) # Unique was unavailable on Pytorch 0.3, it is now :)
                        w = par.unique() # In pytorch 1.0
                        for i in range(len(w)):
                            '''
                            This section needs to be optimized... 
                            This is really slow.
                            If someone has an idea, please let me know to learn more :) 
                            '''
                            mask = par==w[i]
                            grads = par.grad[mask]
                            new_shared_grad = grads.sum()
                            par.grad[mask] = new_shared_grad.repeat(len(grads))
            self.optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += outputs.max(1)[1].eq(targets.data).cpu().sum().item()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        return train_loss/(batch_idx+1), 100.*correct/total
