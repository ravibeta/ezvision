export type MessageType = "user" | "bot";

export interface Message {
  id: string;
  type: MessageType;
  text: string;
  imageUrl?: string;
  downloadUrl?: string;
}
