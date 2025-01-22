def run_step(optimizer, g_constraint, model, cond, data, label, loss_fn):
    optimizer.g_constraint = g_constraint
    
    optimizer.zero_grad()

    label_pred = model(data, None)
    loss_uncond = loss_fn(label_pred, label)
    loss_uncond.backward()
    optimizer.g_value = loss_uncond.item()
    optimizer.first_step(zero_grad = True)

    label_pred = model(data, cond)
    loss_cond = loss_fn(label_pred, label)
    loss_cond.backward()
    optimizer.second_step()

    optimizer.base_optimizer.step()