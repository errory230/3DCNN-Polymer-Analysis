device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 512
LEARNING_RATE =0.0001
NUM_EPOCHS = 30
INPUT_CHANNELS = 5 

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn_with_permute)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn_with_permute)


model = VoxelPredictor(input_channels=INPUT_CHANNELS).to(device)

criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n--- Starting Training ---")
best_val_loss = float('inf')
best_epoch = -1
for epoch in range(NUM_EPOCHS):
    model.train() 
    train_loss = 0.0
    for batch_idx, (voxel1_batch, voxel2_batch, meta_info_batch, target_batch, target_id) in enumerate(train_loader):
        voxel1_batch = voxel1_batch.to(device)
        voxel2_batch = voxel2_batch.to(device)
        meta_info_batch = meta_info_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()

        outputs = model(voxel1_batch, voxel2_batch, meta_info_batch)
        loss = criterion(outputs, target_batch)

        loss.backward() 
        optimizer.step()

        train_loss += loss.item() * voxel1_batch.size(0) 

    train_loss /= len(train_loader.dataset)

     model.eval() 
    val_loss = 0.0
    with torch.no_grad(): 
        for voxel1_batch, voxel2_batch, meta_info_batch, target_batch, target_id in val_loader:
            voxel1_batch = voxel1_batch.to(device)
            voxel2_batch = voxel2_batch.to(device)
            meta_info_batch = meta_info_batch.to(device)
            target_batch = target_batch.to(device)

            outputs = model(voxel1_batch, voxel2_batch, meta_info_batch)
            loss = criterion(outputs, target_batch)
            val_loss += loss.item() * voxel1_batch.size(0)

        val_loss /= len(val_loader.dataset)

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_epoch = epoch + 1
    
      torch.save(model.state_dict(), "best_voxel_predictor_model_bin.pth")
      print(f"  --> Model saved (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")

print("\n--- Training Complete ---")
print(f"Best model saved from Epoch {best_epoch} with Validation Loss: {best_val_loss:.4f}")

print("--- Training Complete ---")
