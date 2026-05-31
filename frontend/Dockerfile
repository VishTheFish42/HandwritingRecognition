FROM node:20-slim

WORKDIR /app

# Install deps before copying source so this layer is cached
COPY package.json package-lock.json* ./
RUN npm install

COPY . .

EXPOSE 5173

# --host exposes the dev server outside the container
CMD ["npm", "run", "dev", "--", "--host"]
