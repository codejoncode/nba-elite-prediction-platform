python init_db.py             ← Create sample data first
python app.py


What happens:
  ✅ App starts
  ✅ Database initializes automatically
  ✅ New fields added (no data loss)
  ✅ Server runs

No manual cleanup needed.
No data loss.
No extra commands.


═══════════════════════════════════════════════════════════════════════════════
WHY THIS WORKS
═══════════════════════════════════════════════════════════════════════════════

When you run: python app.py

The app does this automatically:

  1. Starts Flask
  2. Calls init_database()
  3. Checks if database exists
  4. Creates all tables (if missing)
  5. Creates default metrics (if missing)
  6. Starts server

All without having to run: rm nba_predictor.db
Done! 🚀

# If issues with node js backend occurs 
#run the following commands
# python diagnose.py 

