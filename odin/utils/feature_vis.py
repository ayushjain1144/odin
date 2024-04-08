import torch
import torch.nn.functional as F

def calculate_principal_components(embeddings, num_components=3):
    
     """Calculates the principal components given the embedding features.
     Args:
         embeddings: A 2-D float tensor of shape `[num_pixels, embedding_dims]`.
         num_components: An integer indicates the number of principal
         components to return.
     Returns:
         A 2-D float tensor of shape `[num_pixels, num_components]`.
     """
     embeddings = embeddings - torch.mean(embeddings, 0, keepdim=True)
     _, _, v = torch.svd(embeddings)
     return v[:, :num_components]

def pca(embeddings, num_components=3, principal_components=None):
    """Conducts principal component analysis on the embedding features.
    This function is used to reduce the dimensionality of the embedding.
    Args:
        embeddings: An N-D float tensor with shape with the
        last dimension as `embedding_dim`.
        num_components: The number of principal components.
        principal_components: A 2-D float tensor used to convert the
        embedding features to PCA'ed space, also known as the U matrix
        from SVD. If not given, this function will calculate the
        principal_components given inputs.
    Returns:
        A N-D float tensor with the last dimension as  `num_components`.
    """
    shape = embeddings.shape
    embeddings = embeddings.view(-1, shape[-1])
    if principal_components is None:
        principal_components = calculate_principal_components(
            embeddings, num_components)
    embeddings = embeddings - torch.mean(embeddings, 0, keepdim=True)
    embeddings = torch.mm(embeddings, principal_components)
    new_shape = list(shape[:-1]) + [num_components]
    embeddings = embeddings.view(new_shape)
    return embeddings

def embedding_to_rgb(embeddings, project_type='pca'):
    """Project high-dimension embeddings to RGB colors.
    Args:
        embeddings: A 4-D float tensor with shape
        `[batch_size, embedding_dim, height, width]`.
        project_type: pca | random.
    Returns:
        An N-D float tensor with shape `[batch_size, 3, height, width]`.
    """
    # Transform NCHW to NHWC.
    embeddings = embeddings.permute(0, 2, 3, 1).contiguous()
    embeddings = F.normalize(embeddings, dim=-1)
    N, H, W, C= embeddings.shape
    if project_type == 'pca':
        rgb = pca(embeddings, 3)
    elif project_type == 'random':
        random_inds = torch.randint(0,
                                    C,
                                    (3,),
                                    dtype=torch.long,
                                    device=embeddings.device)
        rgb = torch.index_select(embeddings, -1, random_inds)
    else:
        raise NotImplementedError()
    # Normalize per image.
    rgb = rgb.view(N, -1, 3)
    rgb -= torch.min(rgb, 1, keepdim=True)[0]
    rgb /= torch.max(rgb, 1, keepdim=True)[0]
    rgb *= 255
    rgb = rgb.byte()
    # Transform NHWC to NCHW.
    rgb = rgb.view(N, H, W, 3)
    rgb = rgb.permute(0, 3, 1, 2).contiguous()
    return rgb

def embedding_to_3d_color(
    embeddings, project_type='pca', principal_components=None):
    """Project high-dimension embeddings to RGB colors.
    Args:
        embeddings: A 3-D float tensor with shape
        `[batch_size, embedding_dim, num_points]`.
        project_type: pca | random.
    Returns:
        An N-D float tensor with shape `[batch_size, 3, points]`.
    """
    # Transform NCP to NPC.
    embeddings = embeddings.permute(0, 2, 1).contiguous() # bs, num_points, embedding_dim
    embeddings = embeddings - torch.mean(embeddings, 1, keepdim=True)
    embeddings = F.normalize(embeddings, dim=-1)
    N, P, C= embeddings.shape
    if project_type == 'pca':
        rgb = pca(embeddings, 3, principal_components)
    elif project_type == 'random':
        random_inds = torch.randint(0,
                                    C,
                                    (3,),
                                    dtype=torch.long,
                                    device=embeddings.device)
        rgb = torch.index_select(embeddings, -1, random_inds)
    else:
        raise NotImplementedError()
    # Normalize per image.
    rgb = rgb.view(N, -1, 3)
    rgb -= torch.min(rgb, 1, keepdim=True)[0]
    rgb /= torch.max(rgb, 1, keepdim=True)[0]
    rgb *= 255
    # rgb = rgb.byte()
    # Transform NHWC to NCHW.
    rgb = rgb.view(N, P, 3)
    rgb = rgb.permute(0, 2, 1).contiguous()
    return rgb

def calculate_prototypes_from_labels(embeddings,
                                    labels,
                                    max_label=None):
    """Calculates prototypes from labels.
    This function calculates prototypes (mean direction) from embedding
    features for each label. This function is also used as the m-step in
    k-means clustering.
    Args:
        embeddings: A 2-D or 4-D float tensor with feature embedding in the
        last dimension (embedding_dim).
        labels: An N-D long label map for each embedding pixel.
        max_label: The maximum value of the label map. Calculated on-the-fly
        if not specified.
    Returns:
        A 2-D float tensor with shape `[num_prototypes, embedding_dim]`.
    """
    embeddings = embeddings.view(-1, embeddings.shape[-1])
    if max_label is None:
        max_label = labels.max() + 1
    prototypes = torch.zeros((max_label, embeddings.shape[-1]),
                            dtype=embeddings.dtype,
                            device=embeddings.device)
    labels = labels.view(-1, 1).expand(-1, embeddings.shape[-1])
    prototypes = prototypes.scatter_add_(0, labels, embeddings)
    prototypes = F.normalize(prototypes)
    return prototypes