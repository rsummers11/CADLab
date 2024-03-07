# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# Ricardo Bigolin Lanfredi - 2023-14-12
# Description:
# definitions of model architecture and losses
import torch
import torchvision
import types
import sys
import clip_chexzero as clip
from list_labels import str_labels_location, list_of_location_labels_per_abnormality, str_labels_mimic

def forward_inference_ce(out, normalize_fn):
    return torch.sigmoid(out)

def compute_uncertain_regression_loss(out, target):
    mu = out[...,0]
    sigma = out[...,1]
    # mu = torch.nn.functional.softplus(mu)
    # mu = -torch.log(mu)
    # target = -torch.log(torch.clamp(target,0.01,0.99))
    normal_dist = torch.distributions.Normal(mu, torch.nn.functional.softplus(sigma))
    neg_log_likelihood = -normal_dist.log_prob(target)
    return torch.mean(neg_log_likelihood)

class loss_ce(object):
    def __init__(self, args):
        self.criterion_bce = torch.nn.BCEWithLogitsLoss()
        self.criterion_regression = torch.nn.MSELoss()
        self.criterion_regression_uncertain = compute_uncertain_regression_loss
        self.criterion_multiclass = torch.nn.CrossEntropyLoss(label_smoothing=args.severity_smoothing)
        self.label_smoothing = args.label_smoothing
        self.presence_loss_type = args.presence_loss_type
        self.use_hard_labels = args.use_hard_labels
        self.severity_loss_multiplier = args.severity_loss_multiplier
        self.severity_loss_type = args.severity_loss_type
        self.severity_smoothing = args.severity_smoothing
        self.location_smoothing = args.location_smoothing
        self.location_loss_multiplier = args.location_loss_multiplier
        self.stable_probability = args.stable_probability
        self.indices_allowed_each_location = []
        self.uncertainty_label = args.uncertainty_label
        for index_class in range(args.num_classes):
            location_labels_allowed = list_of_location_labels_per_abnormality[str_labels_mimic[index_class]]
            # self.indices_allowed_each_location.append([str_labels_location.index(allowed_location) for allowed_location in location_labels_allowed])
            self.indices_allowed_each_location.append(torch.tensor([ location in location_labels_allowed for location in str_labels_location]).bool())

    def __call__(self, out, labels, severities, location_labels, location_vector_index, probabilities, unchanged_uncertainties):
        total_loss = []
        for index_class in range(len(out)):
            out_labels = out[index_class][0]
            if self.use_hard_labels:
                labels_to_use = labels
                labels_to_use = labels_to_use[:,index_class][:,None]
                labels_to_use[labels_to_use==-3] = self.uncertainty_label
                labels_to_use[labels_to_use==-2] = 0
                labels_to_use[labels_to_use==-1] = self.uncertainty_label
                smoothed_labels = labels_to_use*(1-self.label_smoothing)+(1-labels_to_use)*self.label_smoothing
            else:
                probabilities[probabilities==101] = self.stable_probability 
                smoothed_labels = (probabilities[:,index_class][:,None]/100)*(1-2*self.label_smoothing)+self.label_smoothing
            indice_classification_present = (1-unchanged_uncertainties[:,index_class][:,None]).bool()[:,0]
            if self.presence_loss_type == 'reg':
                criterion = self.criterion_regression
            elif self.presence_loss_type == 'regunc':
                # double outputs
                criterion = self.criterion_regression_uncertain
            elif self.presence_loss_type == 'ce':
                criterion = self.criterion_bce
            
            classification_loss = criterion(out_labels[indice_classification_present,:].float(), smoothed_labels[indice_classification_present,:].float())

            if self.severity_loss_multiplier:
                labels_to_use = labels
                labels_to_use = labels_to_use[:,index_class][:,None]
                severity_targets = severities[:,index_class][:,None]
                severity_targets[severity_targets==0] = -1
                severity_targets[labels_to_use==0] = 0
                
                indices_severity_present = ((1-unchanged_uncertainties[:,index_class][:,None])*(severity_targets>=0)).bool()[:,0]
                if (indices_severity_present*1).sum()>0:
                    if self.severity_loss_type == 'reg':
                        # 1 output
                        criterion = self.criterion_regression
                        n_severity_outputs = 1
                        severity_targets = severity_targets.float()
                        severity_targets = severity_targets[indices_severity_present,:]
                    elif self.severity_loss_type == 'regunc':
                        # 2 outputs
                        criterion = self.criterion_regression_uncertain
                        n_severity_outputs = 2
                        severity_targets = severity_targets.float()
                        severity_targets = severity_targets[indices_severity_present,:]
                    elif self.severity_loss_type == 'ce':
                        #4 outputs
                        criterion = self.criterion_multiclass
                        n_severity_outputs = 4
                        severity_targets = severity_targets.long()[:,0]
                        severity_targets = severity_targets[indices_severity_present]
                    out_severity = out[index_class][2]
                    severity_loss = criterion(out_severity[indices_severity_present,:].float(),severity_targets)
                    classification_loss += self.severity_loss_multiplier*severity_loss
            
            if self.location_loss_multiplier:
                location_targets = location_labels[:,index_class]
                # n labels x [batch size, n locations]
                location_labels_allowed = list_of_location_labels_per_abnormality[str_labels_mimic[index_class]]
                out_location = out[index_class][1]
                # location_targets = torch.stack([location_targets[:,str_labels_location.index(allowed_location)] for allowed_location in location_labels_allowed], dim = 1)
                location_targets = location_targets[:,self.indices_allowed_each_location[index_class]]
                indices_location_present =  ((1-unchanged_uncertainties[:,index_class][:,None])*(location_targets>=0)).bool().flatten()
                if (indices_location_present*1).sum()>0:
                    smoothed_labels = location_targets*(1-self.location_smoothing)+(1-location_targets)*self.location_smoothing
                    location_loss = self.criterion_bce(out_location.flatten()[indices_location_present].float(), smoothed_labels.flatten()[indices_location_present].float())
                    classification_loss += self.location_loss_multiplier*location_loss
            total_loss.append(classification_loss)
        return sum(total_loss)/len(total_loss)

def load_clip(model_path, pretrained=False, context_length=77): 
        """
        FUNCTION: load_clip
        ---------------------------------
        """
        device = torch.device("cpu")
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False) 
        try: 
            model.load_state_dict(torch.load(model_path, map_location=device))
        except: 
            print("Argument error. Set pretrained = True.", sys.exc_info()[0])
            raise
        return model

class ClassifierWithHeads(torch.nn.Module):
    def __init__(self, args, original_model):
        super().__init__()
        self.original_model = original_model
        dropout = self.original_model.classifier[0].p
        lastconv_output_channels = self.original_model.classifier[1].in_features
        self.original_model.classifier = torch.nn.ModuleList()
        self.num_classes = args.num_classes
        if args.severity_loss_multiplier:
            if args.severity_loss_type == 'reg':
                n_severity_outputs = 1
            elif args.severity_loss_type == 'regunc':
                n_severity_outputs = 2
            elif args.severity_loss_type == 'ce':
                n_severity_outputs = 4
        else:
            n_severity_outputs = 0
        
        for index_class in range(self.num_classes):
            heads_this_label = torch.nn.ModuleList()
            if args.location_loss_multiplier:
                n_location_outputs = len(list_of_location_labels_per_abnormality[str_labels_mimic[index_class]])
            else:
                n_location_outputs = 0

            if args.share_first_classifier_layer:
                first_classifier_layer = torch.nn.Sequential(
                torch.nn.Dropout(p=dropout),
                torch.nn.Linear(lastconv_output_channels, args.n_hidden_neurons_in_heads),
                torch.nn.SiLU(),
                torch.nn.Dropout(p=dropout))
                heads_this_label.append(torch.nn.Sequential(first_classifier_layer,
                    torch.nn.Linear(args.n_hidden_neurons_in_heads, (1 + (args.presence_loss_type == 'regunc')*1))))
            else:
                heads_this_label.append(torch.nn.Sequential(
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Linear(lastconv_output_channels, args.n_hidden_neurons_in_heads),
                    torch.nn.SiLU(),
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Linear(args.n_hidden_neurons_in_heads, (1 + (args.presence_loss_type == 'regunc')*1))))
            if args.location_loss_multiplier:
                if args.share_first_classifier_layer:
                    heads_this_label.append(torch.nn.Sequential(first_classifier_layer,
                    torch.nn.Linear(args.n_hidden_neurons_in_heads, n_location_outputs)))
                else:
                    heads_this_label.append(torch.nn.Sequential(
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Linear(lastconv_output_channels, args.n_hidden_neurons_in_heads),
                    torch.nn.SiLU(),
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Linear(args.n_hidden_neurons_in_heads, n_location_outputs)))
            else:
                heads_this_label.append(None)
            if args.severity_loss_multiplier:
                if args.share_first_classifier_layer:
                    heads_this_label.append(torch.nn.Sequential(first_classifier_layer,
                    torch.nn.Linear(args.n_hidden_neurons_in_heads, n_severity_outputs)))
                else:
                    heads_this_label.append(torch.nn.Sequential(
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Linear(lastconv_output_channels, args.n_hidden_neurons_in_heads),
                    torch.nn.SiLU(),
                    torch.nn.Dropout(p=dropout),
                    torch.nn.Linear(args.n_hidden_neurons_in_heads, n_severity_outputs)))
            else:
                heads_this_label.append(None)            
            self.original_model.classifier.append(heads_this_label)
    
    def __getattr__(self, attr):
        if attr=='original_model':
            return self.__dict__['_modules']['original_model']
        if attr=='num_classes':
            return self.__dict__['num_classes']
        return getattr(self.__dict__['_modules']['original_model'], attr)

    def forward(self, x):
        x = self.original_model.features(x)

        x = self.original_model.avgpool(x)
        x = torch.flatten(x, 1)
        outputs = []
        for index_class in range(self.num_classes):
            inner_outputs = []
            for index_type_annotation in range(3):
                if self.classifier[index_class][index_type_annotation] is None:
                    inner_outputs.append(None)
                else:
                    inner_outputs.append(self.classifier[index_class][index_type_annotation](x))
            outputs.append(inner_outputs)
        # x = torch.stack(outputs, dim=1)
        return outputs

def get_model(args):
    if args.model=='v2_s':
        model = torchvision.models.efficientnet_v2_s(weights = args.weights)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, args.num_classes)
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True
    if args.model=='v2_m':
        model = torchvision.models.efficientnet_v2_m(weights = args.weights)
        # model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, args.num_classes)
        model = ClassifierWithHeads(args, model)
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True
    if args.model=='v2_l':
        model = torchvision.models.efficientnet_v2_l(weights = args.weights)
        # model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, args.num_classes)
        model = ClassifierWithHeads(args, model)
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True
    if args.model=='mobilenet':
        model = torchvision.models.mobilenet_v3_small(weights = args.weights)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, args.num_classes)
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True
    if args.model=='shufflenet':
        model = torchvision.models.shufflenet_v2_x0_5(weights = args.weights)
        model.fc = torch.nn.Linear(model.fc.in_features, args.num_classes)
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True
    if args.model[0]=='b':
        model_name = f'efficientnet_{args.model}'
        model = torchvision.models.__dict__[model_name](weights = args.weights)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, args.num_classes)
        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            param.requires_grad = True
    if args.model=='swin':
        model = torchvision.models.swin_v2_b(weights = args.weights)
        model.head = torch.nn.Linear(model.head.in_features, args.num_classes)
        for param in model.parameters():
            param.requires_grad = False

        for param in model.head.parameters():
            param.requires_grad = True
    
    if args.model=='chexzero':
        model = load_clip(
            model_path=args.weights.model_path, 
            pretrained=True, 
            context_length=77
        )
        del model.transformer
        del model.token_embedding
        del model.ln_final
        del model.logit_scale
        del model.text_projection
        del model.positional_embedding

        class ClassifierCLIP(torch.nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.original_model = original_model
                self.fc = torch.nn.Linear(512, args.num_classes)

            def forward(self,image):
                embeddings = self.original_model.encode_image(image) 
                return self.fc(embeddings.view([embeddings.shape[0], -1]))
        model = ClassifierCLIP(model)
    if args.model=='xrv':
        import torchxrayvision as xrv
        model = xrv.models.ResNet(weights="resnet50-res512-all")
        del model.op_threshs
        model.model.fc = torch.nn.Linear(model.model.fc.in_features, args.num_classes)
        for param in model.parameters():
            param.requires_grad = False

        for param in model.model.fc.parameters():
            param.requires_grad = True

    return model