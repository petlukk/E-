# Git Setup Instructions for Linux

Run the following commands in your `EA2` directory on your Linux machine:

1. **Initialize Git**
   ```bash
   git init
   ```

2. **Add Files**
   (The `.gitignore` I created will automatically exclude build artifacts like `target/` and `backups/`)
   ```bash
   git add .
   ```

3. **Verify Staged Files (Optional)**
   ```bash
   git status
   ```
   (Ensure `target/` and `backups/` are NOT listed)

4. **Commit**
   ```bash
   git commit -m "Initial commit: Eä v2 Phase 4 complete"
   ```

5. **Rename Branch to Main**
   ```bash
   git branch -M main
   ```

6. **Add Remote**
   ```bash
   git remote add origin https://github.com/petlukk/E-.git
   ```

7. **Push**
   ```bash
   git push -u origin main
   ```

   *Note: If `git push` asks for a password and you have 2FA enabled, you will need to use a Personal Access Token (PAT) as the password.*
