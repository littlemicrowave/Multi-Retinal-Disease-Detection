from utils.cvae import *
from utils.train_eval import train_vae, visualize_reconstructions, training_graphs_vae, sample, save_samples, DataLoader
from utils.losses import elbo_loss

train_dataset = VAEDataset(train_labels, train_images, transform_in=transform_in, transform_out=transform_out)
val_dataset = VAEDataset(val_labels, val_images, transform_in=transform_in, transform_out=transform_out)

encoder = Encoder(latent_channels=32, backbone_path=resnet_dir).to(device)
decoder = Decoder(latent_channels=32).to(device)
'''
parameters = 0

for p in decoder.parameters():
    parameters+= p.numel()

print("Encoder parameters:", parameters)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr = 2e-4, weight_decay=1e-5)

results = train_vae(encoder, decoder, elbo_loss, train_dataset, val_dataset, freeze_encoder_for=10, optimizer=optimizer, kl_warmup_epochs=50, beta_max=1, epochs=450, save_as="task4/cvae")
training_graphs_vae(results, "task4/cvae_tuning")
'''
encoder.load_state_dict(torch.load("task4/cvae_encoder.pt"))
decoder.load_state_dict(torch.load("task4/cvae_decoder.pt"))
visualize_reconstructions(encoder, decoder, val_dataset, device=device, num_samples=4, denorm_input=True)
labels = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1]],dtype=torch.float).to(device)
samples = sample(4, decoder=decoder, labels=labels, temp=1)
save_samples(samples, labels)