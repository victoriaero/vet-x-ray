import torch.nn as nn
import torchvision.models as models

try:
    import torchxrayvision as xrv
    has_xrv = True
except ImportError:
    has_xrv = False

try:
    from eva_x import eva_x_tiny_patch16, eva_x_small_patch16, eva_x_base_patch16
    has_evax = True
except ImportError:
    has_evax = False

class CustomModel(nn.Module):
    def __init__(self, base_model, num_classes=2, use_dropout=False, dropout_rate=0.5):
        super().__init__()
        self.base = base_model
        self.use_dropout = use_dropout

        if hasattr(base_model, "fc"):  # ResNet
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
        elif hasattr(base_model, "classifier"):  # DenseNet
            in_features = base_model.classifier.in_features
            base_model.classifier = nn.Identity()
        elif hasattr(base_model, "head"):  # EVA-X
            in_features = base_model.head.in_features
            base_model.head = nn.Identity()
        else:
            raise ValueError("Modelo base não reconhecido")

        if use_dropout:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, num_classes)
            )
        else:
            self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.base(x)
        return self.classifier(features)

class XRVWrapper(nn.Module):
    def __init__(self, base_model, feature_dim, num_classes=2, use_dropout=False, dropout_rate=0.5):
        super().__init__()
        self.base = base_model
        self.base_type = type(base_model).__name__.lower()

        if use_dropout:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, num_classes)
            )
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        if "densenet" in self.base_type:
            feats = self.base.features(x)
            feats = nn.functional.relu(feats, inplace=True)
            feats = nn.functional.adaptive_avg_pool2d(feats, (1, 1))
            feats = feats.view(feats.size(0), -1)
        else:
            feats = self.base(x)  # para ResNet

        return self.classifier(feats)


def get_model(model_name, pretrained=True, use_dropout=False, dropout_rate=0.5,
              grayscale_input=False, use_torchxrayvision=False,
              xrv_model_name="densenet121-res224-all",
              use_evax=False, eva_x_size="small"):

    if use_evax:
        if not has_evax:
            raise ImportError("EVA‑X não instalado. Execute `pip install eva_x`.")

        if eva_x_size == "tiny":
            base = eva_x_tiny_patch16(pretrained=True)
        elif eva_x_size == "small":
            base = eva_x_small_patch16(pretrained=True)
        elif eva_x_size == "base":
            base = eva_x_base_patch16(pretrained=True)
        else:
            raise ValueError("Tamanho inválido para EVA‑X: escolha 'tiny', 'small' ou 'base'.")

        return CustomModel(base, use_dropout=use_dropout, dropout_rate=dropout_rate)

    if use_torchxrayvision:
        if not has_xrv:
            raise ImportError("TorchXRayVision não instalado. Execute `pip install torchxrayvision`.")

        if xrv_model_name == "resnet50-res512-all":
            xrv_model = xrv.models.ResNet(weights="resnet50-res512-all")
        elif xrv_model_name == "densenet121-res224-all":
            xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
        else:
            raise ValueError(f"Modelo XRV não suportado: {xrv_model_name}")

        xrv_model.op_threshs = None

        if hasattr(xrv_model.model, "classifier"):
            feature_dim = xrv_model.model.classifier.in_features
            xrv_model.model.classifier = nn.Identity()
        elif hasattr(xrv_model.model, "fc"):
            feature_dim = xrv_model.model.fc.in_features
            xrv_model.model.fc = nn.Identity()
        else:
            raise ValueError("Modelo XRV sem atributo de classificação reconhecido.")

        return XRVWrapper(xrv_model.model, feature_dim, use_dropout=use_dropout, dropout_rate=dropout_rate)

    if model_name == "resnet34":
        base_model = models.resnet34(pretrained=pretrained)
    elif model_name == "resnet50":
        base_model = models.resnet50(pretrained=pretrained)
    elif model_name == "densenet121":
        base_model = models.densenet121(pretrained=pretrained)
    else:
        raise ValueError(f"Modelo torchvision não suportado: {model_name}")

    if grayscale_input:
        if hasattr(base_model, "conv1"):
            base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif hasattr(base_model, "features") and hasattr(base_model.features[0], "conv"):
            base_model.features[0].conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    return CustomModel(base_model, use_dropout=use_dropout, dropout_rate=dropout_rate)