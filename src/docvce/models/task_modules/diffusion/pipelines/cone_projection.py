import torch
from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def cone_project_direct(
    robust_classifier_gradients: torch.Tensor,
    target_classifier_gradients: torch.Tensor,
    cone_projection_angle_threshold: float,
) -> torch.Tensor:
    """
    Projects the robust/classifier-free gradient onto the non-robust gradient using
    a direct cone projection method.

    Args:
        gradient_robust_classifier (torch.Tensor): Gradient of the loss w.r.t. the robust/classifier-free model.
        gradient_target_classifier (torch.Tensor): Gradient of the loss w.r.t. the non-robust model.
        projection_angle_threshold_in_degrees (float): Degree threshold for the cone projection.

    Returns:
        torch.Tensor: Projected gradient.
    """

    # convert the projection angle threshold to radians
    radians = torch.tensor(
        [cone_projection_angle_threshold],
        device=robust_classifier_gradients.device,
    ).deg2rad()

    # find normalized gradient angles before projection
    normalized_gradient_angles = torch.acos(
        (robust_classifier_gradients * target_classifier_gradients).sum(1)
        / (
            robust_classifier_gradients.norm(p=2, dim=1)
            * target_classifier_gradients.norm(p=2, dim=1)
        )
    )

    # normalize the gradients
    target_classifier_gradients /= target_classifier_gradients.norm(p=2, dim=1).view(
        robust_classifier_gradients.shape[0], -1
    )

    # compute the cone projection of the robust gradient onto the non-robust gradient
    robust_classifier_gradients = (
        robust_classifier_gradients
        - (
            (robust_classifier_gradients * target_classifier_gradients).sum(1)
            / (target_classifier_gradients.norm(p=2, dim=1) ** 2)
        ).view(robust_classifier_gradients.shape[0], -1)
        * target_classifier_gradients
    )

    # normalize the gradients
    robust_classifier_gradients /= robust_classifier_gradients.norm(p=2, dim=1).view(
        robust_classifier_gradients.shape[0], -1
    )

    # compute the cone projection
    cone_projection = (
        robust_classifier_gradients * torch.tan(radians) + target_classifier_gradients
    )

    # replace all target_classifier_gradients above angle with robust_classifier_gradients
    modified_target_gradients = target_classifier_gradients.clone()
    modified_target_gradients[normalized_gradient_angles > radians] = cone_projection[
        normalized_gradient_angles > radians
    ]
    return modified_target_gradients


@torch.no_grad()
def cone_project_chunked(
    robust_classifier_gradients: torch.Tensor,
    target_classifier_gradients: torch.Tensor,
    cone_projection_angle_threshold: float,
    base_tensor_shape: tuple,
    chunk_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Projects the robust/classifier-free gradient onto the non-robust gradient using
    a chunked projection method.

    Args:
        robust_classifier_gradients (torch.Tensor): Gradient of the loss w.r.t. the robust/classifier-free model.
        target_classifier_gradients (torch.Tensor): Gradient of the loss w.r.t. the non-robust model.
        projection_angle_threshold_in_degrees (float): Degree threshold for the cone projection.
        base_tensor_shape (tuple): The original shape of the gradient tensors.
        chunk_size (int, optional): The size of the chunks. Defaults to 1.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Projected gradient and a mask indicating which chunks were not projected.
    """
    # Convert the projection angle threshold to radians
    radians = torch.tensor(
        [cone_projection_angle_threshold],
        device=robust_classifier_gradients.device,
    ).deg2rad()

    # Chunk the robust classifier gradients
    robust_classifier_chunked_gradients = (
        robust_classifier_gradients.view(*base_tensor_shape)
        .unfold(2, chunk_size, chunk_size)
        .unfold(3, chunk_size, chunk_size)
        .permute(0, 1, 4, 5, 2, 3)
        .reshape(
            base_tensor_shape[0],
            -1,
            base_tensor_shape[-2] // chunk_size,
            base_tensor_shape[-1] // chunk_size,
        )
        .permute(0, 2, 3, 1)
    )

    # Chunk the target classifier gradients
    target_classifier_chunked_gradients = (
        target_classifier_gradients.view(*base_tensor_shape)
        .unfold(2, chunk_size, chunk_size)
        .unfold(3, chunk_size, chunk_size)
        .permute(0, 1, 4, 5, 2, 3)
        .reshape(
            base_tensor_shape[0],
            -1,
            base_tensor_shape[-2] // chunk_size,
            base_tensor_shape[-1] // chunk_size,
        )
        .permute(0, 2, 3, 1)
    )

    # Calculate angles between chunked gradients
    normalized_gradient_angles = torch.acos(
        (robust_classifier_chunked_gradients * target_classifier_chunked_gradients).sum(
            -1
        )
        / (
            robust_classifier_chunked_gradients.norm(p=2, dim=-1)
            * target_classifier_chunked_gradients.norm(p=2, dim=-1)
        )
    )

    # Normalize the target classifier chunked gradients
    target_classifier_chunked_gradients /= target_classifier_chunked_gradients.norm(
        p=2, dim=-1
    ).view(
        robust_classifier_chunked_gradients.shape[0],
        robust_classifier_chunked_gradients.shape[1],
        robust_classifier_chunked_gradients.shape[1],
        -1,
    )

    # Compute the cone projection of the robust gradient onto the non-robust gradient
    robust_classifier_chunked_gradients = (
        robust_classifier_chunked_gradients
        - (
            (
                robust_classifier_chunked_gradients
                * target_classifier_chunked_gradients
            ).sum(-1)
            / (target_classifier_chunked_gradients.norm(p=2, dim=-1) ** 2)
        ).view(
            robust_classifier_chunked_gradients.shape[0],
            robust_classifier_chunked_gradients.shape[1],
            robust_classifier_chunked_gradients.shape[1],
            -1,
        )
        * target_classifier_chunked_gradients
    )

    # Normalize the robust classifier chunked gradients
    robust_classifier_chunked_gradients /= robust_classifier_chunked_gradients.norm(
        p=2, dim=-1
    ).view(
        robust_classifier_chunked_gradients.shape[0],
        robust_classifier_chunked_gradients.shape[1],
        robust_classifier_chunked_gradients.shape[1],
        -1,
    )

    # Compute the cone projection
    cone_projection = (
        target_classifier_chunked_gradients.norm(p=2, dim=-1).unsqueeze(-1)
        * robust_classifier_chunked_gradients
        * torch.tan(radians)
        + target_classifier_chunked_gradients
    )

    # Replace all target_classifier_chunked_gradients above angle with robust_classifier_chunked_gradients
    modified_target_chunked_gradients = (
        target_classifier_chunked_gradients.clone().detach()
    )
    modified_target_chunked_gradients[normalized_gradient_angles > radians] = (
        cone_projection[normalized_gradient_angles > radians]
    )

    # Reshape the modified gradients back to the original shape
    modified_target_gradients = (
        modified_target_chunked_gradients.permute(0, 3, 1, 2)
        .reshape(
            base_tensor_shape[0],
            base_tensor_shape[1],
            chunk_size,
            chunk_size,
            robust_classifier_chunked_gradients.shape[1],
            robust_classifier_chunked_gradients.shape[2],
        )
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(*(base_tensor_shape))
    )

    return modified_target_gradients, ~(normalized_gradient_angles > radians)


@torch.no_grad()
def cone_project_chunked_zero(
    robust_classifier_gradients: torch.Tensor,
    target_classifier_gradients: torch.Tensor,
    cone_projection_angle_threshold: float,
    base_tensor_shape: tuple,
    chunk_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Projects the robust/classifier-free gradient onto the non-robust gradient using
    a chunked zero projection method.

    Args:
        gradient_robust_classifier (torch.Tensor): Gradient of the loss w.r.t. the robust/classifier-free model.
        gradient_target_classifier (torch.Tensor): Gradient of the loss w.r.t. the non-robust model.
        projection_angle_threshold_in_degrees (float): Degree threshold for the cone projection.
        base_tensor_shape (tuple): The original shape of the gradient tensors.
        chunk_size (int, optional): The size of the chunks. Defaults to 1.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Projected gradient and a mask indicating which chunks were not zeroed.
    """
    # convert the projection angle threshold to radians
    radians = torch.tensor(
        [cone_projection_angle_threshold],
        device=robust_classifier_gradients.device,
    ).deg2rad()

    # chunk the gradients into bins
    robust_classifier_chunked_gradients = (
        robust_classifier_gradients.view(*base_tensor_shape)
        .unfold(2, chunk_size, chunk_size)
        .unfold(3, chunk_size, chunk_size)
        .permute(0, 1, 4, 5, 2, 3)
        .reshape(
            base_tensor_shape[0],
            -1,
            base_tensor_shape[-2] // chunk_size,
            base_tensor_shape[-1] // chunk_size,
        )
        .permute(0, 2, 3, 1)
    )

    # chunk the gradients into bins
    target_classifier_chunked_gradients = (
        target_classifier_gradients.view(*base_tensor_shape)
        .unfold(2, chunk_size, chunk_size)
        .unfold(3, chunk_size, chunk_size)
        .permute(0, 1, 4, 5, 2, 3)
        .reshape(
            base_tensor_shape[0],
            -1,
            base_tensor_shape[-2] // chunk_size,
            base_tensor_shape[-1] // chunk_size,
        )
        .permute(0, 2, 3, 1)
    )

    # find normalized gradient angles between chunked gradients
    normalized_gradient_angles = torch.acos(
        (robust_classifier_chunked_gradients * target_classifier_chunked_gradients).sum(
            -1
        )
        / (
            robust_classifier_chunked_gradients.norm(p=2, dim=-1)
            * target_classifier_chunked_gradients.norm(p=2, dim=-1)
        )
    )

    # filter out the gradients that are above the threshold angle
    filtered_gradient_chunked = target_classifier_chunked_gradients.clone().detach()
    filtered_gradient_chunked[normalized_gradient_angles > radians] = 0.0

    # import matplotlib.pyplot as plt

    # # Plot the robust classifier chunked gradients
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 4, 1)
    # plt.imshow(
    #     robust_classifier_chunked_gradients[0].cpu().numpy(),
    #     cmap="viridis",
    # )
    # plt.title("Robust Classifier Chunked Gradients")
    # plt.colorbar()

    # # Plot the target classifier chunked gradients
    # plt.subplot(1, 4, 2)
    # plt.imshow(
    #     target_classifier_chunked_gradients[0].cpu().numpy(),
    #     cmap="viridis",
    # )
    # plt.title("Target Classifier Chunked Gradients")
    # plt.colorbar()

    # # Plot the normalized gradient angles
    # plt.subplot(1, 4, 3)
    # plt.imshow(~(normalized_gradient_angles > radians)[0].cpu().numpy(), cmap="viridis")
    # plt.title("Normalized Gradient Angles")
    # plt.colorbar()

    # # plot the filtered_gradient_chunked_reshaped
    # plt.subplot(1, 4, 4)
    # plt.imshow(filtered_gradient_chunked[0].cpu().numpy(), cmap="viridis")
    # plt.title("Filtered Gradient Chunked Reshaped")
    # plt.colorbar()

    # plt.tight_layout()
    # plt.show()

    filtered_gradient_chunked_reshaped = (
        filtered_gradient_chunked.permute(0, 3, 1, 2)
        .reshape(
            base_tensor_shape[0],
            base_tensor_shape[1],
            chunk_size,
            chunk_size,
            robust_classifier_chunked_gradients.shape[1],
            robust_classifier_chunked_gradients.shape[2],
        )
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(*(base_tensor_shape))
    )

    return filtered_gradient_chunked_reshaped, ~(normalized_gradient_angles > radians)
