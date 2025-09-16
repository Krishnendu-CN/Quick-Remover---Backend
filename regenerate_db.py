from auth import Base, engine
Base.metadata.create_all(bind=engine)
print("Database regenerated with updated schema!")
