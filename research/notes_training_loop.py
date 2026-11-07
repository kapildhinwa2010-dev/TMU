# Training loop - minimal
optimizer = torch.optim.Adam(tmu.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(50):  # ~10 min on T4
    for batch in dataloader:
        inputs, targets = batch
        time_feat = tmu.get_time_features()
        output = model(inputs + tmu.time_encoder(time_feat))
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()